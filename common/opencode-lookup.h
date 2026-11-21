#pragma once

#include "common.h"
#include "opencode-tool-proposals.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// The model sees the unmodified tool result. This index is used only for drafting.
class common_opencode_lookup {
    static constexpr size_t width = 8;

    struct match {
        size_t file;
        size_t start;
    };

    std::vector<llama_tokens>           files;
    std::vector<bool>                   full_previews;
    std::unordered_map<uint64_t, match> positions;

    bool   adaptive         = false;
    bool   has_previous     = false;
    match  previous         = {};
    size_t previous_history = 0;
    size_t previous_length  = 0;

    // Extract only ordinary absolute paths from the most recent user request.
    static std::vector<std::string> user_paths(std::string_view text, size_t before_file) {
        const auto last_user = text.rfind("<|im_start|>user", before_file);
        const auto prev =
            last_user == text.npos || !last_user ? text.npos : text.rfind("<|im_start|>user", last_user - 1);
        const auto               start = prev == text.npos ? (before_file > 12000 ? before_file - 12000 : 0) : prev;
        auto                     span  = text.substr(start, before_file - start);
        std::vector<std::string> paths;
        for (size_t i = 0; i < span.size() && paths.size() < 8; ++i) {
            if (span[i] != '/' || (i > 0 && !std::strchr(" \t\n\r\"'`(", span[i - 1]))) {
                continue;
            }
            const auto  end = span.find_first_of(" \n\r\t\"'`<>(){}[]", i + 1);
            std::string path(span.substr(i, end == span.npos ? end : end - i));
            while (!path.empty() && (path.back() == '.' || path.back() == ',')) {
                path.pop_back();
            }
            if (path.size() > 2 && path.size() < 1024 && path.find('/', 1) != path.npos &&
                std::find(paths.begin(), paths.end(), path) == paths.end()) {
                paths.emplace_back(std::move(path));
            }
            if (end == span.npos) {
                break;
            }
            i = end - 1;
        }
        return paths;
    }

    static uint64_t hash(const llama_token * key) {
        uint64_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < width; ++i) {
            h = (h ^ static_cast<uint32_t>(key[i])) * 1099511628211ULL;
        }
        return h;
    }

    void add_file(const llama_vocab * vocab, const std::string & text, bool specials = false) {
        auto tokens = common_tokenize(vocab, text, false, specials);
        if (tokens.size() <= width) {
            return;
        }
        for (size_t i = 0; i + width < tokens.size(); ++i) {
            positions[hash(tokens.data() + i)] = { files.size(), i };
        }
        files.push_back(std::move(tokens));
        full_previews.push_back(specials);
    }

  public:
    void reset(const llama_vocab * vocab, const llama_tokens & prompt) {
        const char * enabled = std::getenv("LLAMA_OPENCODE_ADAPTIVE");
        adaptive             = enabled && std::strcmp(enabled, "1") == 0;
        has_previous         = false;
        positions.clear();
        files.clear();
        full_previews.clear();
        constexpr std::string_view marker      = "<type>file</type>\n<content>\n";
        const char *               frames      = std::getenv("LLAMA_OPENCODE_PREDICT_EDITS");
        const bool                 full_frames = adaptive && frames && std::strcmp(frames, "1") == 0;
        const auto                 text        = common_detokenize(vocab, prompt, full_frames);
        std::string_view           view(text);
        size_t                     offset = 0;
        while ((offset = view.find(marker, offset)) != std::string_view::npos) {
            const size_t before_file = offset;
            offset += marker.size();
            std::string raw;
            uint64_t    previous = 0;
            while (offset < view.size()) {
                const size_t end    = view.find('\n', offset);
                const auto   line   = view.substr(offset, end == std::string_view::npos ? end : end - offset);
                uint64_t     number = 0;
                const char * last   = line.data() + line.size();
                const auto   parsed = std::from_chars(line.data(), last, number);
                if (parsed.ec != std::errc() || number == 0 || (previous && number != previous + 1) ||
                    last - parsed.ptr < 2 || parsed.ptr[0] != ':' || parsed.ptr[1] != ' ') {
                    break;
                }
                raw.append(parsed.ptr + 2, last);
                raw.push_back('\n');
                previous = number;
                offset   = end == std::string_view::npos ? view.size() : end + 1;
            }
            if (!raw.empty()) {
                add_file(vocab, raw);
                if (adaptive) {
                    const auto path_start = view.rfind("<path>", before_file);
                    const auto path_end   = view.rfind("</path>", before_file);
                    if (path_start != view.npos && path_end != view.npos && path_start + 6 < path_end &&
                        path_end - path_start < 1024) {
                        const std::string source(view.substr(path_start + 6, path_end - path_start - 6));
                        const std::string prefix   = raw.substr(0, 640);
                        const std::string suffix   = raw.substr(raw.size() > 640 ? raw.size() - 640 : 0);
                        const std::string call     = "<tool_call>\n<function=";
                        const std::string path_tag = ">\n<parameter=filePath>\n";
                        const std::string mid      = "\n</parameter>\n<parameter=";
                        const std::string end      = "\n</parameter>\n</function>\n</tool_call>";
                        add_file(vocab, call + "edit" + path_tag + source + mid + "oldString>\n" + prefix);
                        add_file(vocab, suffix + mid + "newString>\n" + prefix);
                        add_file(vocab, suffix + end);
                        const auto paths = user_paths(view, before_file);
                        for (const auto & path : paths) {
                            if (path != source) {
                                add_file(vocab, call + "write" + path_tag + path + mid + "content>\n" + prefix);
                            }
                        }
                        if (full_frames && view.find(marker, offset) == view.npos) {
                            const auto message = opencode_proposals::instruction(view, before_file);
                            for (const auto & candidate : opencode_proposals::tool_calls(message, source, paths, raw)) {
                                add_file(vocab, candidate, true);
                            }
                        }
                    }
                }
            }
        }
    }

    bool propose(const llama_tokens & history, llama_token last, size_t max_tokens, llama_tokens & out) {
        if (positions.empty() || history.size() < width - 1 || !max_tokens) {
            return false;
        }
        llama_token key[width];
        std::copy(history.end() - (width - 1), history.end(), key);
        key[width - 1]   = last;
        const auto found = positions.find(hash(key));
        if (found == positions.end()) {
            return false;
        }
        match  loc      = found->second;
        size_t advanced = 0;
        bool   followed = false;
        if (adaptive && has_previous && history.size() >= previous_history) {
            advanced         = history.size() - previous_history;
            const auto & old = files[previous.file];
            const auto   at  = previous.start + advanced;
            if (advanced <= previous_length + 1 && at < old.size() && old.size() - at > width &&
                std::equal(key, key + width, old.begin() + at)) {
                loc      = { previous.file, at };
                followed = true;
            }
        }
        const auto & file  = files[loc.file];
        const auto   start = file.begin() + loc.start;
        if (!std::equal(key, key + width, start)) {
            return false;
        }
        const auto first   = start + width;
        size_t     horizon = max_tokens;
        if (adaptive) {
            if (!followed) {
                const bool recent_miss = has_previous && history.size() > previous_history &&
                                         history.size() - previous_history < previous_length / 4;
                const size_t first = full_previews[loc.file] && !recent_miss ? 160 : 24;
                horizon            = std::min(max_tokens, first);
            } else if (advanced >= previous_length) {
                horizon = std::min(max_tokens, std::max<size_t>(32, previous_length * 2));
            } else {
                horizon = std::min(max_tokens, std::max<size_t>(12, advanced));
            }
        }
        const size_t count = std::min(horizon, static_cast<size_t>(file.end() - first));
        if (count < 3) {
            return false;
        }
        previous         = loc;
        previous_length  = count;
        previous_history = history.size();
        has_previous     = true;
        out.assign(first, first + count);
        return true;
    }
};
