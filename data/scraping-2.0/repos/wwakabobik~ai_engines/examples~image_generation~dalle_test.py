# -*- coding: utf-8 -*-
"""
Filename: dalle_test.py
Author: Iliya Vereshchagin
Copyright (c) 2023. All rights reserved.

Created: 15.10.2023
Last Modified: 17.10.2023

Description:
This file contains testing procedures for DALLE experiments
"""
import asyncio
from openai_python_api import DALLE

# pylint: disable=import-error
from examples.creds import oai_token, oai_organization  # type: ignore


dalle = DALLE(auth_token=oai_token, organization=oai_organization)


async def main():
    """Main function for testing DALLE."""
    resp = await dalle.create_image_url("robocop (robot policeman, from 80s movie)")
    print(resp)
    resp = await dalle.create_variation_from_url(resp[0])
    print(resp)


asyncio.run(main())
