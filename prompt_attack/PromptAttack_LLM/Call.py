import time
import openai
import json
import sqlite3
import logging
import threading
import hashlib
import requests


class LLMLogSql:
    def __init__(self, log_file) -> None:
        self.log_file = log_file
        conn = sqlite3.connect(log_file)
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS my_table
                  (Q TEXT PRIMARY KEY, V TEXT)"""
        )
        self.lock = threading.Lock()
        self.local = threading.local()

    def get_connection(self):
        # 获取线程本地的数据库连接，如果不存在则创建新连接
        if not hasattr(self.local, "conn"):
            self.local.conn = sqlite3.connect(self.log_file)
        return self.local.conn

    def DBQuery(self, Q):
        # hash_algorithm = hashlib.sha256()
        # hash_algorithm.update(Q.encode("utf-8"))
        # hash_value = hash_algorithm.hexdigest()

        with self.lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT V FROM my_table WHERE Q=?", (Q,))
            result = cursor.fetchone()
        return result[0] if result else None

    def DBInsert(self, Q, V):
        # hash_algorithm = hashlib.sha256()
        # hash_algorithm.update(Q.encode("utf-8"))
        # hash_value = hash_algorithm.hexdigest()

        with self.lock:
            conn = sqlite3.connect(self.log_file)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO my_table (Q, V) VALUES (?, ?)", (Q, V)
            )
            conn.commit()


class LLMCall(LLMLogSql):
    log_count: int = 0
    save_count: int = 0

    def __init__(self, log_file, API_key, API_base, version) -> None:
        super().__init__(log_file)

        self.API_base = API_base
        # openai.api_type = "azure"
        # openai.azure_endpoint = API_base
        # openai.api_key = API_key
        # openai.api_version = version
        #self.client = OpenAI(api_key=self.API_key, base_url=self.API_base)

    def call(self, prompt):
        headers = {'Content-type': 'application/json'}
        payload = {'prompt': prompt}
        while True:
            try:
                response = requests.post(url=self.API_base, 
                            data=json.dumps(payload), 
                            headers=headers)
                if response.status_code == 200:
                    break
            except Exception as e:
                logging.warning(e)
                time.sleep(2)
        return response.json()['response']

    def query(self, prompt):
        if save_response := self.DBQuery(prompt):
            self.log_count = self.log_count + 1
            # print(self.log_count / (self.save_count + self.log_count))
            return save_response
        response = self.call(prompt)
        self.DBInsert(prompt, response)
        self.save_count = self.save_count + 1
        # print(self.log_count / (self.save_count + self.log_count))
        return response
