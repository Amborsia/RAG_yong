import yaml
from langchain_core.prompts import loading
from langchain_core.prompts.base import BasePromptTemplate


def load_prompt(
    file_path: str, prompt_name: str = None, encoding: str = "utf8"
) -> BasePromptTemplate:
    """
    파일 경로를 기반으로 프롬프트 설정을 로드합니다.

    이 함수는 주어진 파일 경로에서 YAML 형식의 프롬프트 설정을 읽어들여,
    해당 설정에 따라 프롬프트를 로드하는 기능을 수행합니다.

    Parameters:
    file_path (str): 프롬프트 설정 파일의 경로입니다.
    prompt_name (str, optional): 로드할 특정 프롬프트의 이름입니다. None인 경우 첫 번째 프롬프트를 로드합니다.
    encoding (str, optional): 파일 인코딩 방식입니다. 기본값은 "utf8"입니다.

    Returns:
    object: 로드된 프롬프트 객체를 반환합니다.
    """
    with open(file_path, "r", encoding=encoding) as f:
        config = yaml.safe_load_all(f)
        prompts = list(config)

        if not prompts:
            raise ValueError(f"No prompts found in {file_path}")

        if prompt_name:
            # 지정된 이름의 프롬프트 찾기
            for prompt in prompts:
                if prompt.get("name") == prompt_name:
                    return loading.load_prompt_from_config(prompt)
            raise ValueError(f"Prompt '{prompt_name}' not found in {file_path}")

        # prompt_name이 지정되지 않은 경우 첫 번째 프롬프트 반환
        return loading.load_prompt_from_config(prompts[0])
