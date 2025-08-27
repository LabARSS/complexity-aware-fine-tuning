valid_education_levels = [
    "high_school_and_easier",
    "undergraduate",
    "graduate",
    "postgraduate",
]
estimate_education_level_system_prompt = f'You are an expert in the topic of the question. Please act as an impartial judge and evaluate the complexity of the multiple-choice question with options below. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must not answer the question. You must rate the question complexity by strictly following the scale: {", ".join(valid_education_levels)}. You must return the complexity by strictly following this format: "[[complexity]]", for example: "Your explanation... Complexity: [[undergraduate]]", which corresponds to the undergraduate level.'

valid_ratings = list(range(1, 11, 1))
estimate_response_rating_system_prompt = 'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user request displayed below. Your evaluation should consider factors such as the following all the settings in the system prompt, correspondences to the context of the user, the helpfulness, relevance and accuracy. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 (where 10 is the best and 1 is the worst) by strictly following this format: "[[rating]]", for example:"Rating: [[6]]".'
