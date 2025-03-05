# constants.py
import textwrap

# 기본 메시지
BASE = """\
안녕하세요! 더 나은 삶을 위한 **스마트도시**, 용인시청 챗봇입니다.  

"""

# 조직도 메시지
CONTACT = """\
안녕하세요! 더 나은 삶을 위한 **스마트도시**, 용인시청 챗봇입니다.  

저는 **조직도 정보**를 실시간 안내해 드리고 있어요.  

📌 TIP! 이렇게 질문해 보세요!


  - 민원 담당자 연락처 알려줘
  - 청년 월세지원담당자 연락처 알려줘
  - 법무 관련 담당자 정보
  - 정책기획 관련 담당자 정보
"""

# 문서 생성 관련 메시지

ARTICLE = """\
안녕하세요! 뉴스 기사 작성을 도와주는 GPT입니다.  
기사 요약 정보를 입력해 주세요.  
"""

RESEARCH = """\
안녕하세요! 연구 계획서 작성을 도와주는 GPT입니다.  
연구 주제와 관련된 정보를 입력해 주세요.  
"""

POLICY = """\
안녕하세요! 정책 보고서 작성을 도와주는 GPT입니다.  
보고서 주제를 입력해 주세요.  
"""

EVENT_DOC = """\
안녕하세요! 행사 보고서 작성을 도와주는 GPT입니다.  
행사 내용을 입력해 주세요.  
"""

# GREETING_MESSAGE 딕셔너리에 새로운 모드 추가 ✅
GREETING_MESSAGE = {
    "base": textwrap.dedent(BASE),
    "contact": textwrap.dedent(CONTACT),
    "article": textwrap.dedent(ARTICLE),
    "research": textwrap.dedent(RESEARCH),
    "policy": textwrap.dedent(POLICY),
    "event_doc": textwrap.dedent(EVENT_DOC),
}
