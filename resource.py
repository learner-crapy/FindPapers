from pydantic import BaseModel, Field


DB_NAME = "dazelu"
DB_USER = ""
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = 5432

class IfHighlyAbout(BaseModel):
    result: bool=Field(description="if the paper is highly about the user's requirements")

USER_PROMPT = '''
Is this paper highly about agent debate systems, and call external knowledge,like external database, training dataset, web data etc to help the agent improve the performance.
If yes, return True, else return False.
'''

KEY_INFO_PROMPT = '''
Please extract the following information:
1. **Methods / Techniques** — What methods, models, or algorithms are proposed or used?
2. **Datasets / Benchmarks** — Which datasets or benchmarks are used for evaluation?
3. **Results / Improvements** — What are the main results and how much do they improve over baselines?
'''