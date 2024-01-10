import re
from os import path, makedirs
from datetime import datetime
from shutil import rmtree
from flask import url_for, redirect, flash, Markup, jsonify
from sqlalchemy import text

from flask_wtf import FlaskForm
from wtforms import HiddenField, StringField, SubmitField, IntegerField, SelectField, BooleanField
from wtforms.validators import Length, ValidationError

from flask_app.models import db, Chat, ChatQuestion, DocSet, DocSetFile, UserGroup, Job
from flask_app.helpers import render_chat_template, size_to_human, upload_file
from flask_app.background_jobs import background_jobs

from langchain.vectorstores import Chroma
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings

'''
This form contains all for inserting, updating and deleting a docset
'''

class DocSetForm(FlaskForm):

    # Field definitions
    id = HiddenField('ID', default=0)
    name = StringField('Name', default='', validators=[Length(min=3, max=64)], render_kw={'size': 40})
    llm_type = SelectField('LLM type', default='chatopenai', choices=['chatopenai', 'huggingface', 'local_llm'])
    llm_modeltype = SelectField('LLM model type', default='gpt35', choices=['gpt35', 'gpt35_16', 'gpt4', 'llama2', 'GoogleFlan'])
    embeddings_provider = SelectField('Embeddings provider', default='openai', choices=['openai', 'hugging_face', 'local_embeddings'])
    embeddings_model = SelectField('Embeddings model', default='text-embedding-ada-002', choices=['text-embedding-ada-002', 'all-mpnet-base-v2'])
    text_splitter_method = SelectField('Text splitter method', default='NLTKTextSplitter', choices=['NLTKTextSplitter', 'RecursiveCharacterTextSplitter'])
    chain = SelectField('Chain', default='conversationalretrievalchain', choices=['conversationalretrievalchain'])
    chain_type = SelectField('Chain type', default='stuff', choices=['stuff'])
    chain_verbosity = BooleanField('Chain verbosity', default=False, render_kw={'class': 'yes-checkbox'})
    search_type = SelectField('Search type', default='similarity', choices=['similarity'])
    vecdb_type = SelectField('Vector database', default='chromadb', choices=['chromadb'])
    chunk_size = IntegerField('Chunk size', default=1000, render_kw={'min': 100, 'max': 5000, 'step': 100})
    chunk_overlap = IntegerField('Chunk overlap', default=200)
    chunk_k = IntegerField('Chunk k', default=4)
    submit = SubmitField('Save')


    # Custom validation    ( See: https://wtforms.readthedocs.io/en/stable/validators/ )
    '''
    # if LLM_TYPE is "chatopenai" then LLM_MODEL_TYPE must be one of: "gpt35", "gpt35_16", "gpt4"
# if LLM_TYPE is "huggingface" then LLM_MODEL_TYPE must be one of "llama2", "GoogleFlan"
# "llama2" requires Huggingface Pro Account and access to the llama2 model https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
# note: llama2 is not fully tested, the last step was not undertaken, because no HF Pro account was available for the developer
# Context window sizes are currently:
# "gpt35": 4097 tokens which is equivalent to ~3000 words
# "gpt35_16": 16385 tokens
# "gpt4": 8192 tokens
# "GoogleFlan": ? tokens
# "llama2": ? tokens
# LLM_MODEL_TYPE must be one of "llama2", "GoogleFlan"
LLM_MODEL_TYPE = "gpt35"
# EMBEDDINGS_PROVIDER must be one of: "openai", "huggingface"
EMBEDDINGS_PROVIDER = "openai"
# EMBEDDINGS_MODEL must be one of: "text-embedding-ada-002", "all-mpnet-base-v2"
# If EMBEDDINGS_MODEL is "all-mpnet-base-v2" then EMBEDDINGS_PROVIDER must be "huggingface"

    '''
    def validate_name(form, field):
        if not re.search(r'^[a-zA-Z0-9-_ ]+$', field.data):
            raise ValidationError('Invalid name; Only letters, digits, spaces, - _ characters allowed.')
        same_docset = DocSet.query.filter(DocSet.name == field.data.strip(), DocSet.id != form.docset_id_for_validation).all()
        if len(same_docset) >= 1:
            raise ValidationError('This name already exists.')

    def validate_llm_modeltype(form, field):
        if form.llm_type.data == 'chatopenai':
            supported = ['gpt35', 'gpt35_16', 'gpt4']
            if not (field.data in supported):
                raise ValidationError('The LLM type \'chatopenai\' only supports model types: ' + ', '.join(supported))
        if form.llm_type.data == 'huggingface':
            supported = ['llama2', 'GoogleFlan']
            if not (field.data in supported):
                raise ValidationError('This LLM type \'huggingface\' only supports model types: ' + ', '.join(supported))

    def validate_embeddings_model(form, field):
        if form.embeddings_provider.data == 'openai':
            supported = ['text-embedding-ada-002']
            if not (field.data in supported):
                raise ValidationError('The embeddings provider \'openai\' only supports embeddings models: ' + ', '.join(supported))
        if form.embeddings_provider.data == 'hugging_face':
            supported = ['all-mpnet-base-v2']
            if not (field.data in supported):
                raise ValidationError('This embeddings provider \'hugging_face\' only supports embeddings models: ' + ', '.join(supported))


    # Handle the request (from routes.py) for this form
    def handle_request(self, method, id, file_id=0):

        usergroups, checked, files = [], [], []

        if id > 0:
            # Get user groups with permission for this docset
            usergroups = db.session.execute(text('SELECT id, name, docset_' + str(id) + ' FROM ' + UserGroup.__tablename__ + ';')).fetchall()
            for usergroup in usergroups:
                if getattr(usergroup, 'docset_' + str(id)) == 1:
                    checked.append('checked="checked"')
                else:
                    checked.append('')

        # Show the form
        if method == 'GET':
            if id > 0:
                # Get record from database and set the form values (if id == 0 the defaults are used)
                obj = DocSet.query.get(id)
                obj.fields_to_form(self)

            # Show the form
            return render_chat_template('docset.html', form=self, id=id, usergroups = usergroups, checked=checked)
        
        # Save the form
        if method == 'POST':
            self.docset_id_for_validation = id
            if self.validate():
                if id >= 1:
                    # The table needs to be updated with the new values
                    obj = DocSet.query.get(id)
                    obj.fields_from_form(self)
                    db.session.commit()

                    self.setUserGroupFields()

                    flash('The document set is saved.', 'info')
                    return redirect(url_for('docsets'))

                # A new record must be inserted in tyhe table
                obj = DocSet()
                obj.fields_from_form(self)
                db.session.add(obj)
                db.session.commit()
                id = obj.id

                makedirs(obj.get_doc_path(), exist_ok=True)
                
                self.setUserGroupFields()

                flash('The document set is saved.', 'info')
                return redirect(url_for('docset', id=id))
            
            # Validation failed: Show the form with the errors
            return render_chat_template('docset.html', form=self, id=id, usergroups = usergroups, checked=checked)

        # Show the chunks from the docset
        if method == 'CHUNKS':
            obj = DocSet.query.get(id)
            obj.fields_to_form(self)
            doc_dir = obj.get_doc_path()
            files, files_ = [], DocSetFile.query.filter(DocSetFile.docset_id == id).order_by(DocSetFile.no).all()
            to_document = ''
            for file in files_:
                file_full_name = path.join(doc_dir, file.filename)
                if path.exists(file_full_name):
                    fdt = datetime.fromtimestamp(path.getctime(file_full_name)).strftime('%d-%m-%Y %H:%M:%S')
                    fsz = size_to_human(path.getsize(file_full_name))
                else:
                    fdt = '- deleted -'
                    fsz = '- deleted -'
                files.append({
                    'id': file.id, 
                    'no': file.no, 
                    'name': file.filename, 
                    'dt': fdt,
                    'size': fsz,
                })
                #f = open(file_full_name, 'r')
                #to_document += f.read().replace('\r\n', '\n')
                #f.close()
            #embeddings = OpenAIEmbeddings()
            vectordb_folder = obj.create_vectordb_name()
            vector_store = Chroma(
                            collection_name=obj.get_collection_name(),
                            # embedding_function=embeddings,
                            persist_directory=vectordb_folder,
                        )
            sources = vector_store.get()
            i, documents, metadatas = 0, sources['documents'], sources['metadatas']
            chunks = []
            for document in documents:
                metadata = metadatas[i]
                l = len(document)
                if l >= 160:
                    l = 160
                chunk1, chunk2, chunk3 = document[:int(l / 2)], document[int(l / 2): int(l / -2)], document[int(l / -2):]
                chunks.append({'chunk1': chunk1, 'chunk2': chunk2, 'chunk3': chunk3, 'metadata': metadata})
                i += 1
            chunks.sort(key=lambda x: 10000 * x['metadata']['file_no'] + x['metadata']['chunk_no'])
            return render_chat_template('docset-chunks.html', form=self, files=files, chunks=chunks)

        # Show the files from the docset
        if method == 'FILES':
            obj = DocSet.query.get(id)
            obj.fields_to_form(self)
            return render_chat_template('docset-files.html', form=self, docset_id=id)

        # Upload a file to the docset
        if method == 'UPLOAD-FILE':
            obj = DocSet.query.get(id)
            return upload_file(obj)

        # Get file statusses for the docset
        if method == 'STATUS':
            obj = DocSet.query.filter(DocSet.id == id).first()
            if obj:
                #files, files_ = [], DocSetFile.query.outerjoin(Job, Job.bind_to_id == DocSetFile.id).with_entities(DocSetFile.id, DocSetFile.no, DocSetFile.filename, Job.status_system, Job.status, Job.status_msg).filter(DocSetFile.docset_id == id).order_by(DocSetFile.no).all()
                files, files_ = [], DocSetFile.query.filter(DocSetFile.docset_id == id).order_by(DocSetFile.no).all()
                if files_:
                    for file in files_:
                        status_msg, status_system, status = '', 'Done', Job.query.filter(Job.bind_to_id == file.id).order_by(Job.id).all()
                        if status:
                            status_msg = status[-1].status_msg
                            status_system = status[-1].status_system
                            status = status[-1].status
                        else:
                            status = 'Done'
                        file_full_name = path.join(obj.get_doc_path(), file.filename)
                        if path.exists(file_full_name):
                            dt = datetime.fromtimestamp(path.getctime(file_full_name)).strftime('%d-%m-%Y %H:%M:%S')
                            sz = size_to_human(path.getsize(file_full_name))
                        else:
                            dt = '- deleted -'
                            sz = '- deleted -'
                        files.append({'id': file.id, 'no': file.no, 'filename': file.filename, 'status_system': status_system, 'status': status, 'status_msg': status_msg, 'dt': dt, 'size': sz})
                status = {'error': False, 'files': files}
            else:
                status = {'error': True, 'msg': 'Document set does not exists (anymore).'}
            return jsonify(status)

        # Delete a file from the docset
        if method == 'DELETE-FILE':
            background_jobs.new_job('Delete file', file_id)
            return jsonify({'start_polling': True}) #redirect(url_for('docsets'))

        # Delete the docset
        if method == 'DELETE':
            obj = DocSet.query.get(id)
            # Delete files
            vectordb_folder_path = obj.create_vectordb_name()
            rmtree(path=obj.get_doc_path(), ignore_errors=True)
            rmtree(path=vectordb_folder_path, ignore_errors=True)
            db.session.execute(DocSetFile.__table__.delete().where(DocSetFile.docset_id == id))
            # Delete chatquestions
            db.session.execute(text('DELETE FROM ' + ChatQuestion.__tablename__ + ' WHERE chat_id IN (SELECT id FROM ' + Chat.__tablename__ + ' WHERE docset_id = ' + str(id) + ');'))
            # Delete chats
            db.session.execute(Chat.__table__.delete().where(Chat.docset_id == id))
            # Delete the object
            db.session.delete(obj)
            db.session.commit()
            self.setUserGroupFields()
            flash('The document set has been deleted.', 'info')
            return redirect(url_for('docsets'))

    # See app/forms/usergroup.py for an explanation
    def setUserGroupFields(self):
        # Make sure that the usergroup table has an auth-field for each docset
        docsets = DocSet.query.all()
        usergroup = db.session.execute(text('SELECT * FROM ' + UserGroup.__tablename__ + ';')).keys()
        if len(usergroup) == 0:
            usergroupfields = []
        else:
            usergroupfields = usergroup._keys
        columnnames = []
        # Find fields that do not exsist (and create them)
        for docset in docsets:
            columnname = 'docset_' + str(docset.id)
            columnnames.append(columnname)
            if columnname not in usergroupfields:
                db.session.execute(text('ALTER TABLE ' + UserGroup.__tablename__ + ' ADD COLUMN ' + columnname + ' INTEGER;'))
        # Find fields that should not exists (and delete them)
        for usergroupfield in usergroupfields:
            if usergroupfield[0:7] == 'docset_' and usergroupfield not in columnnames:
                db.session.execute(text('ALTER TABLE ' + UserGroup.__tablename__ + ' DROP COLUMN ' + usergroupfield + ';'))
        db.session.commit()

