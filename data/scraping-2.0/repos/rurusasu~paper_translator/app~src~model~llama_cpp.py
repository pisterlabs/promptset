from typing import Any, Literal


def create_llama_cpp_model(
    package_name: Literal["llama_index", "langchain"],
    model_url: str | None = None,
    model_path: str | None = None,
    temperature: float = 0.0,
    context_window: int = 4096,
    max_tokens: int = 2048,
) -> Any:
    """
    LlamaCPPモデルを生成する関数

    Args:
        package_name (Literal["llama_index", "langchain"]): パッケージ名
        model_url (str | None): モデルのURL
        model_path (str | None): モデルのパス
        temperature (float): 生成される文章の多様性を調整する温度パラメータ
        context_window (int): コンテキストウィンドウのサイズ

    Returns:
        LlamaCPP: 生成されたLlamaCPPモデル
    """
    # モデルのURLまたはパスを取得する
    if isinstance(model_url, str):
        model_url_or_path = model_url
    elif isinstance(model_path, str):
        model_url_or_path = model_path
    else:
        raise ValueError("Either model_url or model_path must be specified.")

    if package_name == "llama_index":
        model = _create_llama_index_cpp_model(
            model_url=model_url_or_path,
            model_path=model_url_or_path,
            temperature=temperature,
            context_window=context_window,
            max_tokens=max_tokens,
        )
    elif package_name == "langchain":
        model = _create_langchain_cpp_model(
            model_path=model_url_or_path,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(
            f"package_name must be one of ['llama_cpp', 'langchain'], but got {package_name}."
        )
    return model


def _create_llama_index_cpp_model(
    model_url: str | None = None,
    model_path: str | None = None,
    temperature: float = 0.0,
    context_window: int = 4096,
    max_tokens: int = 2048,
) -> Any:
    """
    llama_indexパッケージを使用して、LlamaCPPモデルを生成する関数

    Args:
        model_url (str | None): モデルのURL
        model_path (str | None): モデルのパス
        temperature (float): 生成される文章の多様性を調整する温度パラメータ
        context_window (int): コンテキストウィンドウのサイズ
        max_tokens (int): 生成される文章の最大トークン数

    Returns:
        LlamaCPP: 生成されたLlamaCPPモデル
    """

    from llama_index.llms import LlamaCPP as LlamaIndexCPP

    # GPU を使用する場合の設定
    n_gpu_layers = 40
    n_batch = 16
    n_ctx = 4096

    # モデルのURLまたはパスを取得する
    if isinstance(model_url, str):
        model_url_or_path = model_url
    elif isinstance(model_path, str):
        model_url_or_path = model_path
    else:
        raise ValueError("Either model_url or model_path must be specified.")

    try:
        # LlamaCPPモデルを生成する
        model = LlamaIndexCPP(
            model_url=model_url_or_path,
            model_path=model_url_or_path,
            temperature=temperature,
            max_new_tokens=max_tokens,
            context_window=context_window,
            model_kwargs={
                "n_gpu_layers=": n_gpu_layers,
                "n_batch=": n_batch,
                "n_ctx=": n_ctx,
            },
            verbose=True,
        )
        return model
    except FileNotFoundError as e:
        raise (
            f"File not found error occurred during LlamaCPP model creation: {e}"
        )

    except Exception as e:
        raise (f"Error occurred during LlamaCPP model creation: {e}")


def _create_langchain_cpp_model(
    model_path: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Any:
    """
    langchainパッケージを使用して、LlamaCPPモデルを生成する関数

    Args:
        model_path (str | None): モデルのパス
        temperature (float): 生成される文章の多様性を調整する温度パラメータ

    Returns:
        LlamaCPP: 生成されたLlamaCPPモデル
    """
    from langchain.llms import LlamaCpp as LangchainCPP

    # GPU を使用する場合の設定
    n_gpu_layers = 40
    n_batch = 16
    n_ctx = 4096

    try:
        # LlamaCPPモデルを生成する
        model = LangchainCPP(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            n_batch=n_batch,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=True,
        )
        return model
    except FileNotFoundError as e:
        raise (
            f"File not found error occurred during LlamaCPP model creation: {e}"
        )

    except Exception as e:
        raise (f"Error occurred during LlamaCPP model creation: {e}")
