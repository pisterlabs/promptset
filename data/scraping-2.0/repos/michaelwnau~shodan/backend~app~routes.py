from fastapi import APIRouter, Request, Depends, Response, status, HTTPException
import os
import openai
import logging

router = APIRouter()
_logger = logging.getLogger(__name__)
