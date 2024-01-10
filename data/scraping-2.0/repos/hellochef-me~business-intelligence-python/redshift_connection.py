{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d2bf9604-d710-42fc-b3f6-77f354aa3203",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import sqlalchemy as sa\n",
    "from sqlalchemy import create_engine\n",
    "from sqlalchemy.orm import sessionmaker\n",
    "import openai\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d67f3e15-00f9-40d7-a155-41fa0ce875cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import sqlalchemy as sa\n",
    "from sqlalchemy import create_engine\n",
    "from sqlalchemy.orm import sessionmaker\n",
    "import openai\n",
    "import os\n",
    "import psycopg2\n",
    "\n",
    "REDSHIFT_ENDPOINT = \"172.52.0.186\"\n",
    "REDSHIFT_PORT = \"5439\"\n",
    "REDSHIFT_DBNAME = \"hellochef\"\n",
    "REDSHIFT_USER = \"jairaj\"\n",
    "REDSHIFT_PASSWORD = \"\"\n",
    "\n",
    "conn = psycopg2.connect(\n",
    "    host=REDSHIFT_ENDPOINT,\n",
    "    port=REDSHIFT_PORT,\n",
    "    dbname=REDSHIFT_DBNAME,\n",
    "    user=REDSHIFT_USER,\n",
    "    password=REDSHIFT_PASSWORD\n",
    ")\n",
    "\n",
    "cursor = conn.cursor()\n",
    "\n",
    "create_table_query = \"\"\"\n",
    "CREATE TABLE IF NOT EXISTS csv_upload.Holidays (\n",
    "    startDate DATE,\n",
    "    Holidays INTEGER,\n",
    "    SchoolBreak INTEGER\n",
    ");\n",
    "\n",
    "\"\"\"\n",
    "\n",
    "cursor.execute(create_table_query)\n",
    "\n",
    "def write_dataframe_to_redshift(df, table_name, connection):\n",
    "    cursor = conn.cursor()\n",
    "    for index, row in df.iterrows():\n",
    "        insert_query = f\"\"\"\n",
    "            INSERT INTO {table_name} (user_id, question, answer, category)\n",
    "            VALUES ({row['user_id']}, '{row['question']}', '{row['answer']}', '{row['category']}');\n",
    "        \"\"\"\n",
    "        cursor.execute(insert_query)\n",
    "    conn.commit()\n",
    "    cursor.close()\n",
    "    \n",
    "write_dataframe_to_redshift(df, 'public.voice_of_customer_processed2', conn)\n",
    "\n",
    "conn.commit()\n",
    "cursor.close()\n",
    "conn.close()\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
