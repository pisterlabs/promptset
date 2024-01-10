from datetime import datetime
from typing import AsyncIterable, Awaitable, Callable, List, Optional, Union

import openai
from pydantic import BaseModel
from fastapi import Depends, FastAPI, HTTPException, status, APIRouter

from rubatochat.core.database.models import User, OpenAIKEYS, Session, add_item, delete_item_by_id, select
from rubatochat.api import db_engine
from .auth import get_current_active_user

app = APIRouter()

#根据auth_token获取username, 以后清除函数中的username参数

#TODO: 中间一些函数应该是MODEL的部分，之后移到core中

@app.post("/openaikeys/add", tags=["apikeys"])
async def add_openapikeys(openapikeys:List[str], user: User = Depends(get_current_active_user)):
    """
    user to add their apikeys to database
    """
    username = user.username

    with Session(db_engine) as session:
        stmt = select(User).where(User.username == username)
        user = session.exec(stmt)
        user = user.first()
        if not user:
            raise HTTPException(status_code=404, detail="username not found")

    _user_id = user.id

    for apikey in openapikeys:

        _content = apikey
        _isvalid = True
        _create_at = datetime.utcnow()

        add_item(db_engine, "openaikeys", **{
                "content":_content,
                "user_id":_user_id, 
                "create_at":_create_at,
                "is_valid":_isvalid,
                })
    
    return {"status": "success"}

#copied from ...
#not a api currently 
def is_open_ai_key_valid(openai_api_key) -> bool:
    if not openai_api_key:
        return False
    try:
        openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            api_key=openai_api_key,
        )
    except Exception as e:
        return False
    return True


# def get_userid(user: User = Depends(get_current_active_user)):
#     username = user.username
#     with Session(db_engine) as session:
#         stmt = select(User).where(User.username == username)
#         user = session.exec(stmt)
#         user = user.first()
#         if not user:
#             raise HTTPException(status_code=404, detail="username not found")
#     return user

def get_key_by_id(key_id:int, user:User=Depends(get_current_active_user)):
    
    _user_id = user.id
    with Session(db_engine) as session:
        # 构建查询语句
        stmt = select(OpenAIKEYS).where(OpenAIKEYS.user_id == _user_id).where(OpenAIKEYS.id == key_id)
        # 执行查询
        _key = session.exec(stmt)
        _key = _key.first() #解开迭代器？
        # 返回查询结果列表
    if _key:
        return _key
    else:
        raise HTTPException(status_code=404, detail="select key's rank out of range.")


def get_key_by_rank(rank:int, user:User=Depends(get_current_active_user)):
    
    _user_id = user.id
    with Session(db_engine) as session:
        # 构建查询语句
        stmt = select(OpenAIKEYS).where(OpenAIKEYS.user_id == _user_id).order_by(OpenAIKEYS.id).offset(rank).limit(1)
        # 执行查询
        _key = session.exec(stmt)
        _key = _key.first() #解开迭代器？
        # 返回查询结果列表
    if _key:
        return _key
    else:
        raise HTTPException(status_code=404, detail="select key's rank out of range.")

def update_key_by_id(key_id:int, new_content:str, user:User=Depends(get_current_active_user)):

    _user_id = user.id
    with Session(db_engine) as session:
        # 构建查询语句
        stmt = select(OpenAIKEYS).where(OpenAIKEYS.user_id == _user_id).where(OpenAIKEYS.id == key_id)
        # 执行查询
        _key = session.exec(stmt)
        _key = _key.first() #解开迭代器？
        # 返回查询结果列表
        if _key:
            _key.content = new_content
            session.commit()
            session.refresh(_key)

            return _key
        else:
            raise HTTPException(status_code=404, detail="select key's rank out of range.")

def get_key_count(user:User=Depends(get_current_active_user)):

    _user_id = user.id
    with Session(db_engine) as session:
        # 构建查询语句
        stmt = select(OpenAIKEYS).where(OpenAIKEYS.user_id == _user_id)
        # 执行查询
        _key = session.exec(stmt)
        _key = _key.all() #解开迭代器？
        # 返回查询结果列表
        return len(_key)


@app.get("/openaikeys/count", tags=["apikeys"])
async def get_openapikeys_count(user: User=Depends(get_current_active_user)):
    """
    user to get their apikeys in database
    """
    _count = get_key_count(user)

    return {"status": "success",
            "message":"get key count success",
            "key_type": "openai",
            "count": _count,
            }

@app.get("/openaikeys/check", tags=["apikeys"])
async def check_openapikey(rank:int, user:User=Depends(get_current_active_user)):
    """
    user to get their apikeys in database
    """
    _key = get_key_by_rank(rank, user)

    return {"status": "success",
            "key_id": _key.id,
            "key": _key.content,
            "key_obj":_key}


@app.get("/openaikeys/isvalid", tags=["apikeys"])
async def check_openapikey_valid(key_id:int, user: User=Depends(get_current_active_user)):
    """
    user to get their apikeys in database
    """

    raise HTTPException(status_code=403, detail="not implemented")
    #still buggy
    _key = get_key_by_id(key_id, user=user)

    return {"status": "success",
            "key_id": _key.id,
            "is_valid": is_open_ai_key_valid(_key.content),
            }


@app.post("/openaikeys/update", tags=["apikeys"])
async def update_openapikey(key_id:int, new_key:str, user: User=Depends(get_current_active_user)):
    """
    user to update their apikeys in database
    """

    _key = update_key_by_id(key_id,new_content=new_key, user=user)

    return {"status": "success",
            "key_id": _key.id,
            "key": _key.content,
            "key_obj":_key}
    

@app.delete("/openaikeys/delete", tags=["apikeys"])
async def delete_openapikey(key_id:int, user: User=Depends(get_current_active_user)):
    """
    user to delete their apikeys in database
    """
    #BUG: 好像会删掉库里的其他id，必须要有正确的用户筛选
    try:
        delete_item_by_id(db_engine, "openaikeys", key_id)
    except:
        raise HTTPException(status_code=403, detail=f"failed to delete key, key id: {key_id}")

    #如何pop删掉的key？
    return {"status": "success",
            "key_id": key_id,
            "message":"halo"
            }

