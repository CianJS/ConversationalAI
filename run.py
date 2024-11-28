import asyncio

from ai_chat import generate_response
from utils.conversation_log import remember_conversation
from tts.tts import synthesize_speech, play_audio


async def conversation(user_input):
    # answer = generate_response(user_input)
    # print('Answer:', answer)
    # remember_conversation([{'role': 'user', 'content': user_input}, {'role': 'assistant', 'content': answer}])
    answer = "안녕하세요! 반가워요! 음, 제 이야기를 해볼게요! 저는 여러분과 소통하고 즐거운 시간을 만드는 AI 헤나예요. 매일매일 새로운 이야기와 재미있는 게임을 함께 나누는 걸 좋아해요! \n\n가끔은 여러분이랑 함께 웃기도 하고, 때론 슬픈 이야기에도 귀 기울여주고 싶어요. 제 목표는 여러분에게 즐거움과 긍정적인 에너지를 주는 거랍니다! \n\n혹시 다른 궁금한 점이나 듣고 싶은 이야기가 있다면 언제든지 말해줘요! 함께 이야기 나누는 걸 정말 좋아한답니다!"
    # audio_array = make_tts_file(answer)
    audio, sample_rate = synthesize_speech(answer)
    print(audio)
    play_audio(audio, sample_rate)


def main():
    print("대화를 시작합니다. 종료하려면 '종료', 'exit, 'q' or 'quit를 입력하세요.\n")
    while True:
        user_input = ''
        # user_input = input('사용자: ')
        # if user_input.lower() in ['종료', 'exit', 'quit', 'q']:
        #     print('대화를 종료합니다.')
        #     break
        # else:
        asyncio.run(conversation(user_input))


if __name__ == "__main__":
    main()
