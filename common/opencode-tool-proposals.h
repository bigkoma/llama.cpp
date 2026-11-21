#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace opencode_proposals {

// Text from the last human message. A tool result has a user-role wrapper in Qwen's template.
inline std::string_view instruction(std::string_view context, size_t limit) {
    constexpr std::string_view marker = "<|im_start|>user\n";
    while (limit > 0) {
        const auto at = context.rfind(marker, limit - 1);
        if (at == context.npos) {
            break;
        }
        const auto first = at + marker.size();
        const auto end   = std::min(limit, context.find("<|im_end|>", first));
        if (first < end) {
            auto message = context.substr(first, end - first);
            if (message.substr(0, 14) != "<tool_response") {
                return message.substr(0, 8192);
            }
        }
        limit = at;
    }
    return {};
}

inline bool space(char ch) {
    return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

inline std::string_view strip_left(std::string_view text) {
    while (!text.empty() && space(text.front())) {
        text.remove_prefix(1);
    }
    return text;
}

inline std::string_view take_literal(std::string_view & text) {
    text = strip_left(text);
    if (text.empty()) {
        return {};
    }
    const char delim = text.front();
    if (delim == '\'' || delim == '"' || delim == '`') {
        const auto end = text.find(delim, 1);
        if (end == text.npos || end <= 1 || end > 512) {
            return {};
        }
        auto result = text.substr(1, end - 1);
        text.remove_prefix(end + 1);
        return result;
    }
    const auto end = text.find_first_not_of("0123456789");
    if (end == 0) {
        return {};
    }
    auto result = text.substr(0, end);
    text.remove_prefix(result.size());
    return result;
}

// A call or label mentioned explicitly in the instruction can suggest a new argument list.
// Restrict this to unique, single-line occurrences with the same name and matching delimiters.
inline void update_named_groups(std::string & source, std::string_view message) {
    for (size_t pos = 0; pos < message.size(); ++pos) {
        const char open = message[pos];
        if (open != '(' && open != '[') {
            continue;
        }
        const char close = open == '(' ? ')' : ']';
        const auto end   = message.find(close, pos + 1);
        if (end == message.npos || end - pos > 96 ||
            message.substr(pos + 1, end - pos - 1).find_first_of("\n()[]") != message.npos) {
            continue;
        }
        size_t name_end = pos;
        while (name_end > 0 && message[name_end - 1] == ' ') {
            --name_end;
        }
        size_t first = name_end;
        while (first > 0 &&
               (std::isalnum(static_cast<unsigned char>(message[first - 1])) || message[first - 1] == '_')) {
            --first;
        }
        if (name_end - first < 2) {
            continue;
        }
        auto       anchor = message.substr(first, pos - first + 1);
        const auto at     = source.find(anchor);
        if (at == source.npos || source.find(anchor, at + anchor.size()) != source.npos) {
            continue;
        }
        const auto old_end = source.find(close, at + anchor.size());
        if (old_end == source.npos || old_end - (at + anchor.size()) > 96 ||
            source.substr(at + anchor.size(), old_end - (at + anchor.size())).find_first_of("\n()[]") != source.npos) {
            continue;
        }
        source.replace(at, old_end - at + 1, message.substr(first, end - first + 1));
        pos = end;
    }
}

inline std::string preview(std::string_view message, const std::string & source) {
    std::string changed = source;
    for (auto word : { std::string_view("replace "), std::string_view("from ") }) {
        const auto begin = message.find(word);
        if (begin == message.npos) {
            continue;
        }
        auto tail = message.substr(begin + word.size());
        if (tail.substr(0, 8) == "exactly ") {
            tail.remove_prefix(8);
        }
        auto old = take_literal(tail);
        tail     = strip_left(tail);
        if (tail.substr(0, 5) == "with ") {
            tail.remove_prefix(5);
        } else if (tail.substr(0, 3) == "to ") {
            tail.remove_prefix(3);
        } else {
            continue;
        }
        auto newer = take_literal(tail);
        if (old.empty() || newer.empty() || old == newer) {
            continue;
        }
        const auto at = changed.find(old);
        if (at == changed.npos || changed.find(old, at + old.size()) != changed.npos) {
            continue;
        }
        changed.replace(at, old.size(), newer);
    }
    update_named_groups(changed, message);
    return changed;
}

inline std::string_view basename(std::string_view path) {
    const auto pos = path.rfind('/');
    return path.substr(pos == path.npos ? 0 : pos + 1);
}

inline std::string fork_content(const std::string & text, std::string_view source, std::string_view dest) {
    const auto old    = basename(source);
    const auto newer  = basename(dest);
    auto       result = text;
    if (old.empty() || old == newer) {
        return result;
    }
    size_t at = 0;
    while ((at = result.find(old, at)) != result.npos) {
        result.replace(at, old.size(), newer);
        at += newer.size();
    }
    return result;
}

inline std::vector<std::string> tool_calls(std::string_view                 message,
                                           const std::string &              path,
                                           const std::vector<std::string> & candidates,
                                           const std::string &              source) {
    if (message.empty() || source.size() > 32000) {
        return {};
    }
    const std::string        start    = "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=";
    const std::string        path_tag = ">\n<parameter=filePath>\n";
    const std::string        next     = "\n</parameter>\n<parameter=";
    const std::string        end      = "\n</parameter>\n</function>\n</tool_call>";
    std::vector<std::string> result;
    std::string              edited = preview(message, source);
    const auto               ext    = path.rfind('.');
    for (const auto & dest : candidates) {
        if (dest == path || ext == path.npos || dest.size() < path.size() - ext ||
            dest.compare(dest.size() - (path.size() - ext), path.size() - ext, path, ext, path.size() - ext) != 0) {
            continue;
        }
        result.push_back(start + "write" + path_tag + dest + next + "content>\n" + fork_content(edited, path, dest) +
                         end);
        if (result.size() == 2) {
            break;
        }
    }
    std::string old = source;
    if (!old.empty() && old.back() == '\n') {
        old.pop_back();
    }
    if (!edited.empty() && edited.back() == '\n') {
        edited.pop_back();
    }
    result.push_back(start + "edit" + path_tag + path + next + "oldString>\n" + old + next + "newString>\n" + edited +
                     end);
    // The index keeps the latest position for a key. A distinct destination makes write
    // the first proposal for shared call syntax; edit is still reachable by its own prefix.
    std::reverse(result.begin(), result.end());
    return result;
}

}  // namespace opencode_proposals
