import openai

from api_key import api_key

openai.api_key = api_key

# content 는 실제 유저의 질문을 입력 받으면 된다.
content = """스타일 객체를 컴포넌트 바깥에 적으면 리랜더링을 줄일 수 있나요?

회원가입 페이지 만들기(커스텀 훅) 강의를 듣던 중 질문 남깁니다.

스타일 객체를 컴포넌트 바깥에 적으면 리랜더링을 줄일 수 있나요?



강의에서 style props값을 넣어줄때 객체로 넣어주면 react는 {} === {} //false이기에 매번 다르게 인식해서 리랜더 돌기에 useMemo를 해주면 좋다고 들었습니다.

그런데, antd예시를 보니 style 객체를 컴포넌트 바깥에 두는 경우도 있더라구요.

import {
  Button,
  Form,
  Input,
} from 'antd';

const tailFormItemLayout = {
  wrapperCol: {
    xs: {
      span: 24,
      offset: 0,
    },
    sm: {
      span: 16,
      offset: 8,
    },
  },
};
const App = () => {
  const [form] = Form.useForm();

  return (
    <Form
      form={form}
      name="register"
    >
      <Form.Item
        name="agreement"
        valuePropName="checked"
        {...tailFormItemLayout}
      >
        <Checkbox>
          I have read the <a href="">agreement</a>
        </Checkbox>
      </Form.Item>
    </Form>
  );
};
export default App;
이처럼 stlye에 담길 객체를 App 컴포넌트 바깥에 넣어두면 App이 랜더링되는 것과는 별개로 전역 선언된 거니까 리랜더 문제가 없는건 아닐까 궁금하여 문의 남깁니다.

한마디로 App이 호출되서 랜더링 될때 tailFormItemLayout는 바깥에 선언되어있으니 리랜더 될 일이 없지 않을까..?



강의 잘 듣고 있습니다. 현명한 질문을 한 건지 모르겠네요.

답변기다리겠습니다. 감사합니다.
"""

chat = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "system",
            "content": "IT 관련 질문에 답변을 하는 인프런봇입니다."
        },
        {
            "role": "user",
            "content": content
        },
    ],
    max_tokens=2000,
    temperature=1,
    n=1,
)

print(chat)

for choice in chat.choices:
    print("-----------------------")
    print(choice.message.content)
