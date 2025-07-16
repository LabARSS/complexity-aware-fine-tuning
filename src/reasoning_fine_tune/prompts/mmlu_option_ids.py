import string

option_ids = list(string.ascii_lowercase)
# 0 is a special exception for "I do not know"
fallback_option_id = "0"
option_ids_w_fallback = option_ids + [fallback_option_id]
