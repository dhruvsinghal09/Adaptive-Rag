from pydantic import BaseModel


class RouteIdentifier(BaseModel):
    route: str
