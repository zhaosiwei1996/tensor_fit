import orjson as json
import time
import os
import platform
import requests
import subprocess
import csv
import socket
import mysql.connector


class BaseUtils:
    @staticmethod
    def get_timestamp():
        return int(time.time())

    @staticmethod
    def json_to_string(data):
        return json.dumps(data)

    @staticmethod
    def string_to_json(data):
        return json.loads(data)

    @staticmethod
    def read_file(file_path):
        with open(file_path, 'r') as file:
            return file.read()

    @staticmethod
    def write_file(file_path, content):
        with open(file_path, 'w') as file:
            file.write(content)

    @staticmethod
    def append_to_file(file_path, content):
        with open(file_path, 'a') as file:
            file.write(content)

    @staticmethod
    def get_system_type():
        return platform.system()

    @staticmethod
    def send_post_request(url, data):
        response = requests.post(url, json=data)
        return response.json()

    @staticmethod
    def send_get_request(url, params=None):
        response = requests.get(url, params=params)
        return response.json()

    @staticmethod
    def send_delete_request(url):
        response = requests.delete(url)
        return response.json()

    @staticmethod
    def send_put_request(url, data):
        response = requests.put(url, json=data)
        return response.json()

    @staticmethod
    def get_client_ip():
        return socket.gethostbyname(socket.gethostname())

    @staticmethod
    def execute_command(command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        return {
            'output': output.decode('utf-8'),
            'error': error.decode('utf-8'),
            'returncode': process.returncode
        }

    @staticmethod
    def save_csv_file(file_path, data, headers=None):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            if headers:
                writer.writerow(headers)
            writer.writerows(data)


############################
# mysql tools


class MySQLTool:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None

    def connect(self):
        self.connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        self.cursor = self.connection.cursor(dictionary=True)

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def execute_query(self, query, params=None):
        # print(query)
        self.cursor.execute(query, params)
        result = self.cursor.fetchall()
        return result

    def execute_update(self, query, params=None):
        self.cursor.execute(query, params)
        self.connection.commit()
        affected_rows = self.cursor.rowcount
        return affected_rows

    def insert(self, table, data):
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['%s'] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        self.execute_update(query, tuple(data.values()))
        return self.cursor.lastrowid

    def delete(self, table, condition):
        query = f"DELETE FROM {table} WHERE {condition}"
        return self.execute_update(query)

    def update(self, table, data, condition):
        set_clause = ', '.join([f"{column} = %s" for column in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {condition}"
        return self.execute_update(query, tuple(data.values()))

    def select(self, table, columns='*', condition=None):
        query = f"SELECT {columns} FROM {'`'+table+'`'}"
        if condition:
            query += f" WHERE {condition}"
        print(query)
        return self.execute_query(query)

    def create_table(self, table_name, columns):
        query = f"CREATE TABLE IF NOT EXISTS {'`'+table_name+'`'} ({columns})"
        # return self.cursor.execute(query)
        return self.execute_query(query)

    def insert_many(self, table, data):
        with self.connection.cursor() as cursor:
            placeholders = ', '.join(['%s'] * len(data[0]))
            columns = ', '.join(data[0].keys())
            sql = f"INSERT INTO `{table}` ({columns}) VALUES ({placeholders})"
            cursor.executemany(sql, [tuple(d.values()) for d in data])
        self.connection.commit()
