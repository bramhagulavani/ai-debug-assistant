from backend.app.services.stack_trace_parser import StackTraceParser

parser = StackTraceParser()

# --- Python traceback ---
python_trace = """
Traceback (most recent call last):
  File "main.py", line 10, in main
    result = get_user([], 'john')
  File "main.py", line 4, in get_user
    return users[user_id]['name']
IndexError: list index out of range
"""

r = parser.parse(python_trace)
print("=== Python ===")
print("Language :", r.language)
print("Error    :", r.error_type)
print("Message  :", r.error_message)
print("Line     :", r.error_line)
print("Function :", r.error_function)
print(r.to_prompt_string())

# --- JavaScript traceback ---
js_trace = """
TypeError: Cannot read properties of undefined (reading 'name')
    at getUser (app.js:5:20)
    at main (app.js:10:3)
    at Object.<anonymous> (app.js:13:1)
"""

r2 = parser.parse(js_trace)
print("\n=== JavaScript ===")
print("Language :", r2.language)
print("Error    :", r2.error_type)
print("Line     :", r2.error_line)
print(r2.to_prompt_string())