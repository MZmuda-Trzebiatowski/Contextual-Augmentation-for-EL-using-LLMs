from pydantic import BaseModel


class ELTag(BaseModel):
    text: str
    uri: str


class ELTagExtend(ELTag):
    beginIndex: int
    endIndex: int


class ELTagList(BaseModel):
    tags: list[ELTag]
