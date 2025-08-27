import os
from time import sleep

from mistralai import Mistral, SDKError


class MistralAPIClient:
    def __init__(self) -> None:
        api_keys = os.environ["MISTRAL_API_KEYS"]
        self.model = "mistral-large-2411"

        self.clients = [
            Mistral(
                api_key=api_key,
            )
            for api_key in api_keys.split(",")
        ]

        for client in self.clients:
            # Check API is alive
            chat_response = client.chat.complete(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": "What is the best French cheese?",
                    },
                ],
                max_tokens=10,
            )
            print(chat_response.model, chat_response.choices[0].message.content)

        self.SLEEP_DURATION = 1.2
        if len(self.clients) == 2:
            self.SLEEP_DURATION = 0.5
        if len(self.clients) >= 3:
            self.SLEEP_DURATION = 0.2

        print("Sleep duration:", self.SLEEP_DURATION)

        self.api_limit_hits_by_client_ids = {}
        self.reset_api_limits()

        self.request_id = 0

    def query_model(self, messages):
        return self._repeat_if_hit_api_limit(self._query_model)(messages)

    def reset_api_limits(self) -> None:
        self.api_limit_hits_by_client_ids = {}
        for i in range(len(self.clients)):
            self.api_limit_hits_by_client_ids[i] = 0

    def wait(self, duration=None):
        sleep(duration or self.SLEEP_DURATION)

    def _repeat_if_hit_api_limit(self, f):  # (1)
        def wrapper(*args, **kw):  # (2)
            while True:
                try:
                    return f(*args, **kw)
                except SDKError as e:
                    if e.status_code == 429:
                        client_id = self.request_id % len(self.clients)
                        self.api_limit_hits_by_client_ids[client_id] += 1

                        total_hits = 0
                        for value in self.api_limit_hits_by_client_ids.values():
                            total_hits += value

                        if (total_hits % 10) == 0:
                            print(f"API limit hit {total_hits} times. Details: {self.api_limit_hits_by_client_ids}")
                        self._wait(2)
                    else:
                        raise e
                except Exception as e:
                    print("repeat_if_hit_api_limit -> unknown error", e)
                    self._wait(60)

        return wrapper

    def _query_model(self, messages):
        self.request_id
        # print(request_id % len(clients))
        client = self.clients[self.request_id % len(self.clients)]
        self.request_id += 1
        response = client.chat.complete(model=self.model, messages=messages)
        return response
