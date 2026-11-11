/**
 * Curated diverse typing-practice sentence bank (Korean & English).
 * Every round draws a different sentence without repetition until the deck cycles,
 * matching real-world typing practice ("짧은글 연습" / Monkeytype quotes).
 */

export const SENTENCE_POOLS = {
  ko: [
    '바람이 불어오는 곳으로 천천히 걸어가면 새로운 풍경이 열린다',
    '작은 습관이 모여 결국 한 사람의 고유한 리듬과 결과를 만든다',
    '커피 향기가 번지는 조용한 아침에는 생각이 한결 맑아진다',
    '오늘 걸어간 발자국이 내일의 길을 조금 더 단단하게 비춰준다',
    '깊은 밤 창문을 열면 선선한 공기가 하루의 피로를 씻어낸다',
    '서두르지 않고 꾸준히 나아가는 사람이 결국 가장 멀리 도착한다',
    '익숙한 길도 계절이 바뀌면 전혀 다른 색깔로 우리를 맞이한다',
    '비 온 뒤 맑게 개인 하늘 아래로 선명한 무지개가 떠올랐다',
    '좋은 질문 하나가 백 가지 성급한 대답보다 더 깊은 해답을 준다',
    '손끝에서 이어지는 미세한 박자는 누구도 쉽게 흉내 낼 수 없다',
    '오래된 책장을 넘길 때마다 잊고 지냈던 계절의 기억이 깨어난다',
    '햇살이 나뭇잎 사이로 부서지며 산책길 위에 잔잔한 그림자를 그린다',
    '마음이 복잡할 때는 잠시 멈춰 서서 깊게 숨을 고르는 것이 좋다',
    '함께 나누는 짧은 대화가 때로는 긴 하루를 버티게 하는 힘이 된다',
    '파도가 밀려왔다 빠져나간 모래사장 위로 하얀 거품이 반짝인다',
    '새로운 기술을 배운다는 것은 세상을 바라보는 창문을 넓히는 일이다',
    '차분한 저녁 음악 소리가 방 안의 공기를 따뜻하게 채워주었다',
    '정성껏 적어 내려간 한 줄의 문장에는 그 사람의 마음이 담겨 있다',
    '가을바람이 스쳐 지나간 자리마다 노란 은행잎이 수북이 쌓였다',
    '작은 호기심에서 출발한 시도가 예상치 못한 멋진 발견으로 이어진다',
    '새벽 공기를 가르며 달리는 기차 창밖으로 안개 낀 들판이 스쳐간다',
    '같은 문장을 타이핑해도 사람마다 손가락이 머무는 호흡은 모두 다르다',
    '맑은 시냇물이 바위를 돌아 흐르듯 자연스러운 흐름이 가장 강하다',
    '별빛이 쏟아지는 고요한 언덕 위에서 내일의 새로운 계획을 세웠다',
  ],
  en: [
    'every person types with a subtle rhythm that fingers remember naturally',
    'quiet mornings with warm coffee bring clarity to complicated thoughts',
    'small daily habits shape the long arc of everything we build over time',
    'walking along the river at sunset makes the busy city feel peaceful',
    'steady curiosity opens doors that rushing forward would easily miss',
    'fresh rain on summer pavement leaves a clean scent in the evening air',
    'good design removes the noise so only what truly matters stays clear',
    'gentle waves rolled across the sandy shore under a bright silver moon',
    'learning how things work beneath the surface turns doubt into confidence',
    'even when two people write the exact same words their cadence differs',
    'autumn leaves drifted across the quiet campus path in the cool breeze',
    'patience and careful practice turn awkward motion into effortless flow',
    'a single clear question often reveals more than a hundred quick guesses',
    'sunlight filtered through tall pine trees along the winding mountain trail',
    'music playing softly in the background helps the mind settle into focus',
    'great journeys begin the moment we decide to take one honest step forward',
    'old notebooks hold quiet snapshots of ideas that once felt brand new',
    'snow covered the rooftops overnight and turned the whole street silent',
    'listening closely to others teaches us perspectives we never imagined',
    'simple tools built with care tend to outlast complicated inventions',
    'distant train whistles echoed softly across the misty valley at dawn',
    'writing thoughts down on paper helps untangle even the hardest problems',
    'bright stars appeared one by one as twilight faded into deep indigo sky',
    'kind words spoken at the right moment stay in memory for many years',
  ],
};

export class SentenceDeck {
  constructor() {
    this.queues = { ko: [], en: [] };
    this.lastPicked = { ko: '', en: '' };
  }

  next(lang = 'ko') {
    const key = lang === 'en' ? 'en' : 'ko';
    if (!this.queues[key] || this.queues[key].length === 0) {
      const copy = [...SENTENCE_POOLS[key]];
      for (let i = copy.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [copy[i], copy[j]] = [copy[j], copy[i]];
      }
      // Avoid boundary repeat across deck reshuffles
      if (copy.length > 1 && copy[0] === this.lastPicked[key]) {
        [copy[0], copy[1]] = [copy[1], copy[0]];
      }
      this.queues[key] = copy;
    }
    const picked = this.queues[key].shift();
    this.lastPicked[key] = picked;
    return picked;
  }
}
