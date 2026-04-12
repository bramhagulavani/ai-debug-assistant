from dotenv import load_dotenv
load_dotenv()

from backend.app.services.ast_parser import ASTParserService

svc = ASTParserService()
print("Backend:", svc.backend_name)

code = """
import os

class UserManager:
    def get_user(self, users, user_id):
        return users[user_id]['name']

def main():
    mgr = UserManager()
    print(mgr.get_user([], 'john'))
"""

ctx = svc.parse(code=code, error_line=5, filename="app.py")
print(ctx.to_prompt_string())