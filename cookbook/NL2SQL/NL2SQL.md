# Yi-Coder: A Powerful Natural Language to SQL Converter

Welcome to the Yi-Coder tutorial! This document will guide beginners on how to use Yi-Coder for natural language to SQL conversion. Yi-Coder is a powerful tool that can understand natural language queries and translate them into accurate SQL statements. In this experiment, we will be using the Yi-Coder-9B-Chat model, a large language model optimized for handling complex database query tasks.

## 1. Project Overview

Our project comprises the following key components:

1. **NL2SQLConverter:** Responsible for converting natural language into SQL queries.
2. **DatabaseManager:** Manage the creation, data insertion, and query execution of an SQLite database.
3. **Main Function:** Orchestrate the entire process.

Let's walk through the implementation of each component step by step.

## 2. Environment Setup

First, we need to import the necessary libraries and set up logging. If any packages are missing, simply install them using the command `pip install <>`.

```python
import re
import sqlparse
import sqlite3
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

This code imports all the required libraries and sets up basic logging configuration. Logging will help us track the execution flow of our program.


## 3. Data Structure Definitions

Next, we define two data classes to represent the database schema and SQL queries.

```python
@dataclass
class DatabaseSchema:
    tables: List[str]

@dataclass
class SQLQuery:
    raw: str
    formatted: str
```

The `DatabaseSchema` class is used to store information about the database tables, while the `SQLQuery` class stores the raw and formatted SQL queries.

## 4. NL2SQLConverter Class

This is the core class of our project, responsible for converting natural language to SQL queries.

```python
class NL2SQLConverter:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").eval()

    def preprocess(self, input_text: str) -> str:
        return re.sub(r'[^\w\s]', '', input_text)

    def process_with_large_model(self, input_text: str, schema_info: DatabaseSchema) -> str:
        prompt = self._construct_prompt(input_text, schema_info)
        messages = self._construct_messages(prompt)
        model_inputs = self._prepare_model_inputs(messages)
        return self._generate_response(model_inputs)

    def _construct_prompt(self, input_text: str, schema_info: DatabaseSchema) -> str:
        return f"""
        Given the following database schema:
        {schema_info.tables}

        Convert the following natural language query to SQL:
        {input_text}

        Please provide only the SQL query without any additional explanation.
        """

    def _construct_messages(self, prompt: str) -> List[dict]:
        return [
            {"role": "system", "content": "You are a helpful assistant that converts natural language to SQL."},
            {"role": "user", "content": prompt}
        ]

    def _prepare_model_inputs(self, messages: List[dict]) -> Any:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer([text], return_tensors="pt").to(self.device)

    def _generate_response(self, model_inputs: Any) -> str:
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            eos_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def nl2sql(self, input_text: str, schema_info: DatabaseSchema) -> SQLQuery:
        cleaned_text = self.preprocess(input_text)
        logger.info(f"Cleaned text: {cleaned_text}")

        raw_sql = self.process_with_large_model(cleaned_text, schema_info)
        formatted_sql = self.postprocess(raw_sql)

        return SQLQuery(raw=raw_sql, formatted=formatted_sql)

    def postprocess(self, sql_query: str) -> str:
        return sqlparse.format(sql_query, reindent=True, keyword_case='upper')

```

This class contains the following key methods:

- `__init__`: Initializes the model and tokenizer.
- `preprocess`: Cleans the input text.
- `process_with_large_model`: Processes the input using the large language model.
- `_construct_prompt`:  Constructs the prompt for the language model, including schema and query.
- `_construct_messages`: Formats the prompt into a message structure for the model.
- `_prepare_model_inputs`: Prepares the input for the model, including tokenization and moving to the correct device.
- `_generate_response`: Generates the SQL response from the model.
- `nl2sql`: The main conversion method that orchestrates the conversion process.
- `postprocess`: Formats the generated SQL query for readability.

## 5. DatabaseManager Class

This class manages our SQLite database.

```python
class DatabaseManager:
    def __init__(self, db_path: str = 'ecommerce.db'):
        self.db_path = db_path

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            self._create_tables(cursor)
            self._insert_sample_data(cursor)

    def _create_tables(self, cursor: sqlite3.Cursor):
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users
        (id INTEGER PRIMARY KEY, username TEXT, email TEXT, registration_date TEXT)
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS products
        (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER)
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders
        (id INTEGER PRIMARY KEY, user_id INTEGER, order_date TEXT, status TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id))
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS order_items
        (id INTEGER PRIMARY KEY, order_id INTEGER, product_id INTEGER, quantity INTEGER,
        FOREIGN KEY (order_id) REFERENCES orders(id),
        FOREIGN KEY (product_id) REFERENCES products(id))
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews
        (id INTEGER PRIMARY KEY, user_id INTEGER, product_id INTEGER, rating INTEGER, comment TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id),
        FOREIGN KEY (product_id) REFERENCES products(id))
        ''')

    def _insert_sample_data(self, cursor: sqlite3.Cursor):
        # Insert users
        users = [
            ('john_doe', 'john@example.com', '2023-01-15'),
            ('jane_smith', 'jane@example.com', '2023-02-20'),
            ('bob_johnson', 'bob@example.com', '2023-03-10'),
            ('alice_brown', 'alice@example.com', '2023-04-05'),
            ('charlie_davis', 'charlie@example.com', '2023-05-12')
        ]
        cursor.executemany('INSERT OR REPLACE INTO users (username, email, registration_date) VALUES (?, ?, ?)', users)

        # Insert products
        products = [
            ('Laptop', 'Electronics', 999.99, 50),
            ('Smartphone', 'Electronics', 599.99, 100),
            ('Running Shoes', 'Sports', 79.99, 200),
            ('Coffee Maker', 'Home Appliances', 49.99, 75),
            ('Book: Python Programming', 'Books', 29.99, 150)
        ]
        cursor.executemany('INSERT OR REPLACE INTO products (name, category, price, stock) VALUES (?, ?, ?, ?)', products)

        # Insert orders
        orders = [
            (1, '2023-06-01', 'Delivered'),
            (2, '2023-06-15', 'Shipped'),
            (3, '2023-07-01', 'Processing'),
            (4, '2023-07-10', 'Delivered'),
            (5, '2023-07-20', 'Shipped')
        ]
        cursor.executemany('INSERT OR REPLACE INTO orders (user_id, order_date, status) VALUES (?, ?, ?)', orders)

        # Insert order items
        order_items = [
            (1, 1, 1),
            (1, 3, 2),
            (2, 2, 1),
            (3, 4, 1),
            (4, 5, 3),
            (5, 1, 1),
            (5, 2, 1)
        ]
        cursor.executemany('INSERT OR REPLACE INTO order_items (order_id, product_id, quantity) VALUES (?, ?, ?)', order_items)

        # Insert reviews
        reviews = [
            (1, 1, 5, 'Great laptop, very fast!'),
            (2, 2, 4, 'Good phone, but battery life could be better'),
            (3, 3, 5, 'Very comfortable shoes'),
            (4, 4, 3, 'Decent coffee maker'),
            (5, 5, 5, 'Excellent book for learning Python')
        ]
        cursor.executemany('INSERT OR REPLACE INTO reviews (user_id, product_id, rating, comment) VALUES (?, ?, ?, ?)', reviews)

    def execute_query(self, sql_query: str) -> Optional[List[Tuple]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                results = cursor.fetchall()

                if not results:
                    logger.warning("No results found.")

                return results
        except sqlite3.Error as e:
            logger.error(f"An error occurred: {e}")
            return None

```

The key methods of this class include:

- `__init__`: Initialize the database path.
- `init_database`: Initialize the database, creating tables and inserting sample data.
- `_create_tables`: Create the necessary tables for the e-commerce database.
- `_insert_sample_data`: Populate the tables with sample data.
- `execute_query`: Execute a given SQL query and returns the results.


## 6. Main Function

Finally, our main function brings all the components together and runs test cases.

```python
def main():
    db_manager = DatabaseManager()
    db_manager.init_database()

    model_path = '01-ai/Yi-Coder-9B-Chat'  # Replace with the actual model path
    converter = NL2SQLConverter(model_path)

    schema_info = DatabaseSchema(tables=[
        "1. users (id, username, email, registration_date)",
        "2. products (id, name, category, price, stock)",
        "3. orders (id, user_id, order_date, status)",
        "4. order_items (id, order_id, product_id, quantity)",
        "5. reviews (id, user_id, product_id, rating, comment)"
    ])

    test_cases = [
        "Show me the top 3 best-selling products",
        "List all users who have made a purchase in the last month",
        "What is the average rating for products in the Electronics category?",
        "Show me the total revenue for each product category",
        "Who are the top 5 users with the most orders?"
    ]

    for case in test_cases:
        sql_query = converter.nl2sql(case, schema_info)
        query_results = db_manager.execute_query(sql_query.formatted)
        logger.info('-' * 50)
        logger.info(f"Query: {case}")
        logger.info(f"Final SQL:\n{sql_query.formatted}")
        logger.info(f"Query Results:\n{query_results}")

if __name__ == "__main__":
    main()
```

The main function performs the following tasks:

1. Initializes the database manager and NL2SQL converter.
2. Defines the database schema and test cases.
3. For each test case, performs the conversion and query, and logs the results.

## Conclusion

Through this project, we demonstrated the capabilities of the Yi-Coder-9B-Chat model in handling complex database queries. It can accurately translate natural language queries into SQL statements and perform well in various complex scenarios. Remember to replace `<Huggingface>` with the actual path to the Yi-Coder-9B-Chat model. This will allow you to run the code and test the functionality of the NL2SQL converter.
