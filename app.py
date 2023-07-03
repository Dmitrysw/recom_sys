import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger
from pydantic import BaseModel

app = FastAPI()

class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH
def load_features():
    #Уникальные записи post_id, user_id где был лайк,
    #чтобы исключить эти посты из рекомендаций
    logger.info("loading liked posts")
    liked_posts_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        where action= 'like' """
    liked_posts = batch_load_sql(liked_posts_query)

    #Фичи по постам
    logger.info("loading posts features")
    posts_features = pd.read_sql(
        """ SELECT * FROM public.posts_info_features_shat""",
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml")
    
    #Фичи по пользователям
    logger.info("loading user features")
    user_features = pd.read_sql(
        """ SELECT * FROM public.user_data""",
        con="postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
            "postgres.lab.karpov.courses:6432/startml")
    
    return [liked_posts, posts_features, user_features]

def load_models():

    model_path = get_model_path(path = "./catboost_model")
    from_file = CatBoostClassifier()
    from_file.load_model(model_path)
    return from_file

logger.info("loading models")
model = load_models()
logger.info("loading features")
features = load_features()
logger.info("service is running")
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id:int, time: datetime, limit: int = 5):
    logger.info(f"user_id: {id}")
    logger.info("reading features")
    user_features = features[2].loc[features[2].user_id == id]
    user_features = user_features.drop("user_id", axis=1)

    #загрузим фичи постов
    logger.info("dropping columns")
    posts_features = features[1].drop(["index", "text"], axis = 1)
    content = features[1][["post_id", "text", "topic"]]

    #Объединим эти фичи
    logger.info("concat all")
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    logger.info("assign everything")
    user_posts_features = posts_features.assign(**add_user_features)
    logger.info("set user_id as index")
    user_posts_features = user_posts_features.set_index('post_id')

    #Добавим инфу о времени рекомендации
    logger.info("add time features")
    user_posts_features["hour"] = time.hour
    user_posts_features["month"] = time.month

    #сформируем предсказанные вероятности для всех постов быть лайкнутыми
    logger.info("predicting")
    predicts = model.predict_proba(user_posts_features)[:, 1]
    user_posts_features["predicts"] = predicts

    #уберем записи, где уже был лайк
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    filtered_ = user_posts_features[~user_posts_features.index.isin(liked_posts)]

    recommended_posts = filtered_.sort_values("predicts")[-limit:].index

    return [
        PostGet(**{
            "id": i,
            "text": content[content.post_id == i].text.values[0],
            "topic": content[content.post_id == i].topic.values[0]
        }) for i in recommended_posts]
    