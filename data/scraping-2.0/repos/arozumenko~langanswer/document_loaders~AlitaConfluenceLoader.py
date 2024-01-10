from langchain.document_loaders import ConfluenceLoader
from langchain.document_loaders.confluence import ContentFormat

class AlitaConfluenceLoader(ConfluenceLoader):
    def load(**kwargs):
        content_formant = kwargs.get('content_format', 'view').lower()
        mapping = {
            'view': ContentFormat.VIEW,
            'storage': ContentFormat.STORAGE,
            'export_view': ContentFormat.EXPORT_VIEW,
            'editor': ContentFormat.EDITOR,
            'anonymous': ContentFormat.ANONYMOUS_EXPORT_VIEW
        }
        kwargs['content_format'] = mapping.get(content_formant, ContentFormat.VIEW)
        return super().load(**kwargs)