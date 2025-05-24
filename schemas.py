from pydantic import BaseModel

class LogMessage(BaseModel):
    message: str
