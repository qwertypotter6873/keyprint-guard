/**
 * Korean and English typing prompt catalog (128 prompts per language: 96 general + 32 pangrams)
 * and persistent anti-repetition SentenceDeck.
 *
 * Anti-repetition guarantees on a device:
 *   1. Persists deck state in `localStorage` (with `sessionStorage` and in-memory fallbacks) and
 *      syncs revisions across tabs so reloads, new tabs, multiple profile enrollments, and repeated
 *      blind identification rounds continue through the 128-sentence deck without repeating.
 *   2. Interleaves general sentences (3/4) and pangrams (1/4) in randomized 4-prompt buckets so
 *      every 5-sentence profile enrollment covers full keyboard geometry without back-to-back pangram fatigue.
 *   3. Maintains a 48-sentence sliding cooldown across deck reshuffles so sentences typed near the
 *      end of one 128-sentence cycle cannot reappear at the start of the next cycle.
 */

export const GENERAL_SENTENCES = {
  ko: [
    '바람이 불어오는 곳으로 천천히 걸어가면 새로운 풍경이 열린다.',
    '작은 습관이 모여 결국 한 사람의 고유한 리듬과 결과를 만든다.',
    '커피 향기가 번지는 조용한 아침에는 생각이 한결 맑아진다.',
    '오늘 걸어간 발자국이 내일의 길을 조금 더 단단하게 비춰 준다.',
    '깊은 밤 창문을 열면 선선한 공기가 하루의 피로를 씻어 낸다.',
    '서두르지 않고 꾸준히 나아가는 사람이 결국 가장 멀리 도착한다.',
    '익숙한 길도 계절이 바뀌면 전혀 다른 색깔로 우리를 맞이한다.',
    '비 온 뒤 맑게 개인 하늘 아래로 선명한 무지개가 떠올랐다.',
    '좋은 질문 하나가 백 가지 성급한 대답보다 더 깊은 해답을 준다.',
    '손끝에서 이어지는 미세한 박자는 누구도 쉽게 흉내 낼 수 없다.',
    '오래된 책장을 넘길 때마다 잊고 지냈던 계절의 기억이 깨어난다.',
    '햇살이 나뭇잎 사이로 부서지며 산책길 위에 잔잔한 그림자를 그린다.',
    '마음이 복잡할 때는 잠시 멈춰 서서 깊게 숨을 고르는 것이 좋다.',
    '함께 나누는 짧은 대화가 때로는 긴 하루를 버티게 하는 힘이 된다.',
    '파도가 밀려왔다 빠져나간 모래사장 위로 하얀 거품이 반짝인다.',
    '새로운 기술을 배운다는 것은 세상을 바라보는 창문을 넓히는 일이다.',
    '차분한 저녁 음악 소리가 방 안의 공기를 따뜻하게 채워 주었다.',
    '정성껏 적어 내려간 한 줄의 문장에는 그 사람의 마음이 담겨 있다.',
    '가을바람이 스쳐 지나간 자리마다 노란 은행잎이 수북이 쌓였다.',
    '작은 호기심에서 출발한 시도가 예상치 못한 멋진 발견으로 이어진다.',
    '새벽 공기를 가르며 달리는 기차 창밖으로 안개 낀 들판이 스쳐 간다.',
    '같은 문장을 타이핑해도 사람마다 손가락이 머무는 호흡은 모두 다르다.',
    '맑은 시냇물이 바위를 돌아 흐르듯 자연스러운 흐름이 가장 강하다.',
    '별빛이 쏟아지는 고요한 언덕 위에서 내일의 새로운 계획을 세웠다.',
    '지하철 문이 닫히자 아이가 양손으로 노란 손잡이를 꼭 잡았다.',
    '이삿짐 상자마다 방 이름을 적어 두니 짐을 풀기가 한결 수월했다.',
    '도서관 사서는 예약 도서에 대출표를 끼우고 안내대 뒤쪽에 세워 두었다.',
    '사과를 반으로 갈라 보니 씨앗이 작은 별처럼 가운데 모여 있었다.',
    '얕은 물에 발을 담근 백로가 꼼짝 않고 물고기를 기다리고 있었다.',
    '낡은 자전거의 체인에 기름을 바르니 오르막에서도 소리가 줄었다.',
    '길모퉁이 국숫집은 점심시간이 지나도 자리를 기다리는 손님이 많았다.',
    '천문대에서는 해가 지기 전부터 망원경의 방향을 맞추기 시작했다.',
    '분실물 상자 맨 아래에서 잃어버린 줄 알았던 연두색 장갑을 찾았다.',
    '새로 단 커튼을 걷으니 맞은편 건물 사이로 낮은 산등성이가 보였다.',
    '온실 한쪽에 놓인 화분에서는 잎보다 먼저 주홍빛 꽃이 피어났다.',
    '심판이 호루라기를 불자 출발선에 서 있던 선수들이 일제히 뛰었다.',
    '강당 무대 뒤에서는 공연에 쓸 의자를 옮기는 소리가 들려왔다.',
    '종이봉투를 펼쳐 보니 안쪽에 시장 골목의 지도가 인쇄되어 있었다.',
    '나는 초록색 연필로 지도에 오늘 자전거로 지나온 길을 그려 넣었다.',
    '정류장 전광판에 도착 시간이 뜨자 사람들은 버스가 오는 쪽을 보았다.',
    '장화를 신은 아이들이 마당에 고인 물을 피해 징검다리를 건넜다.',
    '빵이 식는 동안 선반을 닦고 내일 쓸 밀가루를 항아리에 담아 두었다.',
    '해가 기울자 해변의 돌담과 낮은 지붕이 서서히 붉은빛을 띠었다.',
    '주전자의 물이 끓는 사이 창가에 널어 둔 젖은 수건을 걷었다.',
    '작은 유리병에 물을 채우고 뿌리가 자라기 시작한 바질을 꽂아 두었다.',
    '운동화를 벗고 들어선 연습실에는 잘 닦인 나무 바닥이 드러나 있었다.',
    '공항 안내원은 짐을 찾는 곳으로 이어지는 통로를 손으로 짚어 주었다.',
    '기타의 가장 가는 줄을 새로 갈고 음이 맞을 때까지 조금씩 감았다.',
    '낮은 철문을 밀고 들어가니 벽돌 창고 안에 목재가 가지런히 놓여 있었다.',
    '실험실의 온도를 두 번 더 재고 날짜와 시각을 표에 함께 기록했다.',
    '정비소 벽에는 오래된 자동차에서 떼어 낸 번호판들이 걸려 있었다.',
    '두꺼운 천으로 뜨거운 냄비의 양쪽 손잡이를 감싸 식탁으로 옮겼다.',
    '그늘을 따라 골목을 돌다가 반려견과 함께 나온 이웃을 우연히 만났다.',
    '늦게 도착한 관객들은 연주가 끝나는 틈에 조용히 빈자리로 이동했다.',
    '여행 가방에서 셔츠를 꺼내자 그 아래에 구겨진 승차권 한 장이 있었다.',
    '도시를 벗어나자 신호등이 드물어지더니 곧 낮은 비닐하우스가 보였다.',
    '탁자에 펼친 도면에 팀원들이 각자 맡은 구역을 색칠해 표시했다.',
    '부엌의 낡은 저울을 꺼내 재료의 무게를 재고 작은 그릇에 나눠 담았다.',
    '예보대로 점심 무렵부터 눈발이 날려 공원 벤치가 희끗희끗해졌다.',
    '박물관 표를 접어 주머니에 넣고 천천히 나선형 계단을 올라갔다.',
    '마른 풀을 한쪽으로 걷어 내자 땅 위로 작은 버섯 무리가 보였다.',
    '코트를 수선하는 동안 길 건너 문구점에서 쓸 만한 공책을 골랐다.',
    '인쇄기의 설정을 바꾸고 처음 나온 종이의 여백과 글자 크기를 살폈다.',
    '화단 가장자리에 나란히 핀 봉숭아 아래로 무당벌레가 기어갔다.',
    '새벽 시장 골목에는 갓 쪄 낸 옥수수 김이 하얗게 피어올랐다.',
    '기차역 대합실 시계가 정각을 알리자 개찰구 앞 줄이 움직였다.',
    '수첩 맨 뒷장에 적어 둔 전화번호 덕분에 연락이 바로 닿았다.',
    '비에 젖은 우산을 현관 옆 통에 세워 두고 마른 수건을 찾았다.',
    '오래된 시계추 소리가 고요한 거실의 공기를 일정하게 채웠다.',
    '산등성이 너머로 구름이 걷히자 멀리 항구의 등대가 드러났다.',
    '목공소 작업대 위에는 대패질한 소나무 조각이 가지런히 놓였다.',
    '여름 소나기가 지나간 운동장 트랙 위로 맑은 바람이 불어왔다.',
    '책상 서랍 안쪽에서 학창 시절에 쓰던 만년필과 잉크병을 찾았다.',
    '강변 자전거 도로를 따라 달리다 나무 그늘 아래에서 숨을 돌렸다.',
    '유리창에 맺힌 물방울이 가로등 불빛을 받아 은은하게 반짝였다.',
    '화방 진열대에는 수채화 물감과 붓이 크기 순서대로 놓여 있었다.',
    '약속 시간보다 일찍 도착해 근처 서점에서 새로 나온 책을 보았다.',
    '돌담길을 따라 걷다 보니 감나무 가지마다 주황빛 열매가 열렸다.',
    '악보를 한 장씩 넘기며 막히는 마디를 천천히 다시 연주해 보았다.',
    '캠핑장 모닥불이 잦아들자 밤하늘의 은하수가 한층 또렷해졌다.',
    '세탁소에서 찾아온 겨울 외투를 옷걸이에 걸어 바람을 쐬어 주었다.',
    '선착장에 닿은 유람선에서 여행객들이 가방을 들고 차례로 내렸다.',
    '다락방 창문을 열자 솔방울이 떨어진 지붕 위로 참새가 날아왔다.',
    '전시장 입구에서 나눠 준 안내지를 보며 관람 순서를 미리 정했다.',
    '따뜻한 보리차를 보온병에 가득 채워 배낭 옆주머니에 넣어 두었다.',
    '골목 끝 제과점에서는 오후 세 시마다 고소한 식빵 냄새가 풍겼다.',
    '텃밭에서 막 따 온 상추와 방울토마토를 찬물에 깨끗이 씻어 냈다.',
    '비행기가 구름층을 통과하자 눈부신 파란 하늘이 창밖에 펼쳐졌다.',
    '우체국 창구에서 소포 무게를 확인한 뒤 주소를 한 번 더 읽었다.',
    '오솔길 양옆으로 억새가 흔들리며 가을 햇빛에 은빛으로 빛났다.',
    '칠판 구석에 적힌 오늘의 일정을 지우고 내일 준비물을 적어 두었다.',
    '도자기 공방 선반마다 유약을 바르기 전의 그릇들이 말라 있었다.',
    '저녁노을이 강물 위에 길게 번지며 다리 기둥 그림자를 늘였다.',
    '열람실 조명을 낮추고 노트북 화면 밝기를 눈에 편하게 조절했다.',
    '눈 내린 아침 골목길에 누군가 먼저 쓸어 놓은 좁은 길이 있었다.',
    '식물원 유리 지붕 아래로 야자수 잎이 넓게 퍼져 그늘을 만들었다.',
  ],
  en: [
    'Every person types with a subtle rhythm that fingers remember naturally.',
    'Quiet mornings with warm coffee bring clarity to complicated thoughts.',
    'Small daily habits shape the long arc of everything we build over time.',
    'Walking along the river at sunset makes the busy city feel peaceful.',
    'Steady curiosity opens doors that rushing forward would easily miss.',
    'Fresh rain on summer pavement leaves a clean scent in the evening air.',
    'Good design removes the noise so only what truly matters stays clear.',
    'Gentle waves rolled across the sandy shore under a bright silver moon.',
    'Learning how things work beneath the surface turns doubt into confidence.',
    'Even when two people write the exact same words, their cadence differs.',
    'Autumn leaves drifted across the quiet campus path in the cool breeze.',
    'Patience and careful practice turn awkward motion into effortless flow.',
    'A single clear question often reveals more than a hundred quick guesses.',
    'Sunlight filtered through tall pine trees along the winding mountain trail.',
    'Music playing softly in the background helps the mind settle into focus.',
    'Great journeys begin the moment we decide to take one honest step forward.',
    'Old notebooks hold quiet snapshots of ideas that once felt brand new.',
    'Snow covered the rooftops overnight and turned the whole street silent.',
    'Listening closely to others teaches us perspectives we never imagined.',
    'Simple tools built with care tend to outlast complicated inventions.',
    'Distant train whistles echoed softly across the misty valley at dawn.',
    'Writing thoughts down on paper helps untangle even the hardest problems.',
    'Bright stars appeared one by one as twilight faded into the deep indigo sky.',
    'Kind words spoken at the right moment stay in memory for many years.',
    'The baker set the warm loaves on a rack by the open kitchen window.',
    'A loose screw in the handle made the old garden gate rattle in the wind.',
    'We sorted the spare buttons by size before fixing the coat for winter.',
    'My bicycle chain ran quietly after I washed the grit from each link.',
    'The bus driver waited for a runner who was still halfway down the block.',
    'A brass bell rang whenever someone opened the door of the bookshop.',
    'The library moved its large print books to a shelf near the entrance.',
    'A heron stood beside the reeds while the river rose over the low rocks.',
    'The green team passed the ball twice before taking a shot from the corner.',
    'Folding the map kept the next stretch of the trail in view as we climbed.',
    'The soup simmered on the stove while we sliced bread for the table.',
    'We dried the wet boots upside down on an old towel in the hallway.',
    'A black cat carried its toy mouse under the couch and fell asleep.',
    'The museum labels explained which stones were used to build the bridge.',
    'Each reading in the notebook had a time and a temperature marked beside it.',
    'The first beans of the season were ready to pick along the garden fence.',
    'Our neighbor grew a row of yellow flowers outside the shared laundry room.',
    'The violinist paused to tighten a loose string before the next song.',
    'The buses took a longer route while the road crew fixed the crossing.',
    'My sister drew tiny boats on the plain paper before wrapping the parcels.',
    'We found a dented tin full of marbles in the back of the old cabinet.',
    'One more coat of paint will cover the marks along the kitchen door.',
    'The roof tiles looked darker after a quick shower swept over the town.',
    'The kite began to climb when we let out a little more line on the hill.',
    'Clean sheets of scrap paper came in handy for sketches of the new shelf.',
    'Several apples fell from the tree as the harvest ladder brushed a branch.',
    'The ferry took extra time at the dock while the crew secured the ramp.',
    'The theater left half the seats empty during the morning rehearsal.',
    'The pencil rolled off the desk and stopped beside the broken floor tile.',
    'We froze the last of the berries to use in pancakes later in the month.',
    'My uncle kept a spare key in the pocket of his faded canvas jacket.',
    'The rain gauge filled to the top before the thunder finally moved away.',
    'The florist tied the stems together with a strip of blue cotton thread.',
    'Painted arrows on the pavement pointed visitors toward the railway fair.',
    'Spreading the tools out on the bench showed which wrench was still missing.',
    'Our table was just far enough from the music to hear everyone speak.',
    'A small sign near the creek showed how high the spring flood had reached.',
    'It took three trips to bring all the boxes up to the second floor.',
    'We counted the cash twice before closing the little market stall.',
    'A wire basket in the hall held the mail for everyone on the top floor.',
    'Early sunlight warmed the wooden floorboards near the balcony door.',
    'The station platform grew quiet once the midnight express pulled away.',
    'We packed extra wool socks and a flashlight before climbing the ridge.',
    'Raindrops slid down the cafe window while jazz played softly inside.',
    'A narrow stone path led from the orchard gate down to the boathouse.',
    'She marked the recipe page with a paper clip and preheated the oven.',
    'Fresh cedar shavings curled across the workbench in the small studio.',
    'Cool evening wind carried the scent of rosemary across the courtyard.',
    'The postman tucked a thick brown envelope inside the brass mail slot.',
    'Ripples spread across the pond whenever a dragonfly skimmed the water.',
    'We arranged the chairs in a wide circle before the afternoon meeting.',
    'Morning fog lifted slowly to reveal fishing boats anchored in the bay.',
    'He labeled each glass jar of dried herbs and placed it in the pantry.',
    'The old clock in the hallway chimed six times as dinner came together.',
    'Footsteps echoed down the marble corridor after the art gallery closed.',
    'Wildflowers lined both sides of the gravel lane beyond the farmhouse.',
    'The potter centered a lump of wet clay before pressing down gently.',
    'Shadows lengthened across the tennis court as the sun dipped westward.',
    'We checked the lantern battery twice before pitching camp by the lake.',
    'A warm gust of air rustled the linen curtains in the upstairs bedroom.',
    'The barista steamed a pitcher of oat milk during the morning rush.',
    'Fallen pine needles softened every step along the shaded forest loop.',
    'He tuned the acoustic guitar by ear while waiting for his bandmates.',
    'Clear mountain water spilled over mossy boulders into a shallow pool.',
    'The archivist wore cotton gloves while turning fragile ledger pages.',
    'Distant church bells rang clearly across the tiled rooftops at noon.',
    'We stacked split firewood neatly beneath the shelter of the back porch.',
    'A striped awning shaded the fruit crates outside the corner grocery.',
    'Morning dew sparkled on the spiderweb stretched between two fence posts.',
    'I sketched the skyline quickly before dark clouds hid the harbor tower.',
    'Warm bread and olive oil arrived first at the small corner trattoria.',
    'Lantern light flickered softly against the painted hull of the canoe.',
  ],
};

export const PANGRAM_SENTENCES = {
  ko: [
    '휴일 저녁 해변 테라스에 앉아 향긋한 유자차와 쿠키 요리를 맛보며 풍경을 찍었다.',
    '새벽 기차표를 배낭에 넣어 필름 카메라와 물통을 들고 조용한 휴양림에 갔다.',
    '교실 창가에 둔 큼직한 유리 화병에 붉은 튤립과 노란 유채꽃 향기가 은은히 퍼졌다.',
    '여름 캠프 첫날 텐트 뒤 화로에 모여 우유와 감자 요리를 나누며 야경을 보았다.',
    '토요일 제과점에서 구운 얇은 야채 파이와 버터 쿠키에 흰 우유를 곁들여 먹었다.',
    '박물관 휴게실 탁자에서 전시 표지와 얇은 안내 엽서를 펼쳐 보며 동선을 체크했다.',
    '고양이 두 마리가 햇볕 드는 마루 담요 위에서 파란 튜브와 큼직한 털뭉치를 굴렸다.',
    '해 질 녘 호숫가 벤치 테이블에 스케치북을 펴고 귤과 따뜻한 우유 요리로 야식을 먹었다.',
    '저녁 축제 천막 테이블에서 따뜻한 호박죽과 향긋한 유자차 요리를 종이컵에 채워 팔았다.',
    '독서실 책상 옆에 얇은 포스트잇과 형광펜을 챙겨 두고 컴퓨터로 바른 과학 용어를 찾았다.',
    '새벽 텃밭에서 큼직한 양파와 상추를 베어 담요 옆에 두고 흙 묻은 유리를 닦았다.',
    '낡은 타자기 키로 얇은 새 엽서 표지에 여름 휴가 추억과 오늘 날짜를 적어 넣었다.',
    '저녁 유리 온실에서 향긋한 새 양치류와 보라색 튤립 묘목 화분을 필름 카메라로 찍었다.',
    '휴일 겨울밤 창틈을 커튼으로 막고 테이블 스탠드 램프를 켠 채 얇은 요약집을 읽었다.',
    '우체국 옆 탁자에서 두툼한 봉투에 기념우표를 붙이고 얇은 휴가 엽서를 체크했다.',
    '휴일 저녁 터미널 매점에서 따뜻한 초코 파운드와 얇은 호두 파이 요리를 봉투에 담았다.',
    '일요일 여름 해변 모래밭에서 아이들이 큼직한 초록 튜브와 파란 공으로 얕은 물결을 탔다.',
    '휴일 미술관 가게에서 풍경 엽서와 노트 표지에 얇은 초록 스케치북까지 골라 배낭에 담았다.',
    '조용한 다락방 테이블에서 새 필름 카메라와 물통을 닦고 얇은 여름 휴가 사진첩을 펼쳤다.',
    '수요일 저녁 음악실 테이블에 남아 새 피아노 키와 첼로 튜너로 미묘한 음향을 맞춰 보았다.',
    '휴일 캠핑장 통나무 식탁에 램프를 켜고 향긋한 버섯 수프와 구운 감자 요리를 채워 담았다.',
    '휴일 공원 벤치 테이블에서 차가운 우유와 쿠키를 먹으며 얇은 새 에세이 표지를 넘겼다.',
    '도서관 열람실 컴퓨터로 필요한 참고 자료를 검색해 얇은 노트에 바른 표기와 출처를 적었다.',
    '가을날 학교 트랙을 뛰며 상쾌한 바람과 은은한 유자나무 꽃향기를 천천히 깊게 맡았다.',
    '눈 덮인 벽난로 곁 담요 위에서 털장갑을 말리며 향긋한 새 우유 코코아와 차를 마셨다.',
    '저녁 야시장 가게에서 잘 익은 복숭아와 귤에 키위까지 담아 튼튼한 표지 상자에 채웠다.',
    '항구 옆 책방 테이블에서 귤차와 쿠키를 먹으며 낡은 야간 항해 좌표 지도를 펴 보았다.',
    '초가을 오케스트라 단원들이 무대 뒤편에 모여 플루트와 튜바의 미묘한 음향을 조율하며 섰다.',
    '봄비 내리는 휴일 골목 카페 처마 밑에서 투명 우산을 털고 얇은 요리 잡지와 책을 샀다.',
    '새 스케이트화를 신발장에 넣고 다음 주 빙상장 교육 시간표를 얇은 칠판에 붙여 두었다.',
    '휴일 시골집 테이블에 대나무 돗자리를 펴고 얇은 수박 화채와 쿠키 요리를 나눠 먹었다.',
    '천문대 교육실 스크린 앞에 띄운 목성과 토성 궤도 표를 보며 얇은 공책에 핵심을 적었다.',
  ],
  en: [
    'Five dozen quartz jugs of hot black tea kept our six sleepy crewmen awake.',
    'My friend wove six big plaid jackets before the freezing squall came.',
    'The quick brown fox jumped lazily over seven mossy logs by a quiet creek.',
    'Pack my red box with five dozen quality jugs of sweet sparkling cider.',
    'Sixty zippers were quickly picked from the woven jute bag by a calm clerk.',
    'A wizard quickly mixed five jugs of bright lemon syrup with fizzy club soda.',
    'Bright vixens jumped quickly over lazy fowl near the old wooden gazebo.',
    'Jack quietly moved six dozen big plush chairs forward into the cozy gym.',
    'Five jumping wizards hexed a quiet black crow near the misty olive grove.',
    'We quickly seized a spare black axle and fixed every jammed gear on the van.',
    'My cozy log cabin stove kept six hikers warm just before a wild blizzard quit.',
    'The jovial explorer froze when six quick badgers scrambled up a rocky path.',
    'Lazy movers packed six big antique jugs of wax into our narrow hallway.',
    'Quaint village chefs mixed five dozen jugs of sweet berry syrup for cozy cakes.',
    'Six big juicy plums from our quiet orchard amazed five sleepy kids walking by.',
    'A deft violinist played six crazy Baroque jigs while the meek crowd froze.',
    'Five brisk joggers zipped past a quiet lynx watching from the mossy bluff.',
    'Six cozy campers quietly watched bright northern lights jump above a dark fjord.',
    'The sharp breeze quickly blew six woven flags above a calm fjord plaza.',
    'Seven anxious kids watched a nimble frog jump crazily over the quiet brook.',
    'My uncle fixed a broken quartz watch and gave six prizes to joyful kids.',
    'The sleepy judge quickly wrote six brief memos on a pad of azure vellum.',
    'Brave divers explored a big sunken quartz wreck just off the calm hazy bay.',
    'Six keen botanists quietly prized waxy leaves from a rare jungle orchid vase.',
    'John watched six plump geese fly briskly through a quiet mountain valley haze.',
    'A quick chef glazed five warm berry tarts and boxed up six juicy lemons.',
    'Twelve keen campers spotted a lazy black fox just beyond the quiet gorge.',
    'The joyful wizard quickly fixed five cracked quartz lamps before gathering dusk.',
    'Six joyful musicians played brisk jazz songs while a quiet evening crowd waved.',
    'My lively puppy jumped to chase six buzzing dragonflies past a quiet wicket.',
    'Five calm scribes quickly indexed a dozen dusty journal pages by twilight.',
    'My cozy bookshop manager shelved six rare quartz atlases just before dawn.',
  ],
};

export const SENTENCE_POOLS = {
  ko: [...GENERAL_SENTENCES.ko, ...PANGRAM_SENTENCES.ko],
  en: [...GENERAL_SENTENCES.en, ...PANGRAM_SENTENCES.en],
};

export const RECENT_COOLDOWN = 48;

const POOL_SETS = {
  ko: new Set(SENTENCE_POOLS.ko),
  en: new Set(SENTENCE_POOLS.en),
};

const PANGRAM_SETS = {
  ko: new Set(PANGRAM_SENTENCES.ko),
  en: new Set(PANGRAM_SENTENCES.en),
};

function getPersistentStorage() {
  for (const key of ['localStorage', 'sessionStorage']) {
    try {
      const candidate = globalThis[key];
      if (candidate) return candidate;
    } catch {
      // Fall through when browser privacy settings block storage access.
    }
  }
  return null;
}

function storageKey(lang) {
  return `keyprint_sentence_deck_v2_${lang}`;
}

function legacyStorageKey(lang) {
  return `keyprint_sentence_deck_v1_${lang}`;
}

function cooldownSize(lang) {
  const len = SENTENCE_POOLS[lang]?.length || 0;
  return Math.min(RECENT_COOLDOWN, Math.max(1, Math.floor(len * 0.375)));
}

function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

/**
 * Shuffle `items` while distributing pangrams evenly across 4-sentence buckets (~3 general + 1 pangram).
 */
function interleavePrompts(items, pangramSet) {
  const generals = [];
  const pangrams = [];
  for (const item of items) {
    if (pangramSet.has(item)) pangrams.push(item);
    else generals.push(item);
  }
  shuffleInPlace(generals);
  shuffleInPlace(pangrams);
  if (pangrams.length === 0) return generals;
  if (generals.length === 0) return pangrams;

  const bucketCount = pangrams.length;
  const base = Math.floor(generals.length / bucketCount);
  const extra = generals.length % bucketCount;
  const order = shuffleInPlace(Array.from({ length: bucketCount }, (_, i) => i));
  const extraSet = new Set(order.slice(0, extra));

  const out = [];
  let cursor = 0;
  let trailingGenerals = 1;
  for (let b = 0; b < bucketCount; b++) {
    const gCount = base + (extraSet.has(b) ? 1 : 0);
    const chunk = generals.slice(cursor, cursor + gCount);
    cursor += gCount;
    const minPos = b > 0 && trailingGenerals === 0 ? Math.min(1, gCount) : 0;
    const maxPos = Math.max(minPos, Math.min(gCount, 4 - trailingGenerals));
    const pos = minPos + Math.floor(Math.random() * (maxPos - minPos + 1));
    chunk.splice(pos, 0, pangrams[b]);
    out.push(...chunk);
    trailingGenerals = gCount - pos;
  }
  return out;
}

/**
 * Build a full 128-sentence cycle where recently dealt prompts are placed in the back partition,
 * guaranteeing at least `pool.length - cooldown + 1` (= 81 or 49) draws before any repeat.
 */
function buildCycleQueue(lang, recent) {
  const pool = SENTENCE_POOLS[lang];
  const pangramSet = PANGRAM_SETS[lang];
  const recentSet = new Set(recent);
  const older = [];
  const cooled = [];
  for (const sentence of pool) {
    if (recentSet.has(sentence)) cooled.push(sentence);
    else older.push(sentence);
  }
  const front = interleavePrompts(older, pangramSet);
  const back = interleavePrompts(cooled, pangramSet);
  const combined = [...front, ...back];
  const lastPicked = recent.at(-1) || '';
  if (combined.length > 1 && lastPicked && combined[0] === lastPicked) {
    const swapIdx = 1 + Math.floor(Math.random() * (combined.length - 1));
    [combined[0], combined[swapIdx]] = [combined[swapIdx], combined[0]];
  }
  return combined;
}

function sanitizeRecent(rawRecent, rawLastPicked, poolSet, maxLen) {
  const out = [];
  const seen = new Set();
  if (Array.isArray(rawRecent)) {
    for (const item of rawRecent) {
      if (typeof item === 'string' && poolSet.has(item) && !seen.has(item)) {
        seen.add(item);
        out.push(item);
      }
    }
  }
  if (typeof rawLastPicked === 'string' && poolSet.has(rawLastPicked)) {
    const existingIdx = out.indexOf(rawLastPicked);
    if (existingIdx >= 0) out.splice(existingIdx, 1);
    out.push(rawLastPicked);
  }
  return out.slice(-maxLen);
}

function parseStoredDeck(raw, lang, { allowQueue = true } = {}) {
  const poolSet = POOL_SETS[lang];
  const maxRecent = cooldownSize(lang);
  const stored = JSON.parse(raw ?? 'null');
  if (!stored || typeof stored !== 'object') {
    return { queue: [], recent: [], lastPicked: '', rev: 0 };
  }
  const recent = sanitizeRecent(stored.recent, stored.lastPicked, poolSet, maxRecent);
  const lastPicked = recent.at(-1) || '';
  const queue = stored.queue;
  const validQueue =
    allowQueue &&
    Boolean(lastPicked) &&
    Array.isArray(queue) &&
    queue.length > 0 &&
    queue.length <= poolSet.size &&
    new Set(queue).size === queue.length &&
    queue.every((sentence) => typeof sentence === 'string' && sentence !== lastPicked && poolSet.has(sentence));
  const rev = Number.isInteger(stored.rev) && stored.rev >= 0 ? stored.rev : 0;
  return {
    queue: validQueue ? [...queue] : [],
    recent,
    lastPicked,
    rev,
  };
}

function restoreDeck(storage, lang) {
  try {
    const v2Raw = storage?.getItem(storageKey(lang));
    if (v2Raw !== null && v2Raw !== undefined) {
      return parseStoredDeck(v2Raw, lang, { allowQueue: true });
    }
    const v1Raw = storage?.getItem(legacyStorageKey(lang));
    if (v1Raw !== null && v1Raw !== undefined) {
      // Preserve recent/lastPicked across v1 -> v2 upgrade while building a fresh 128-prompt queue.
      return parseStoredDeck(v1Raw, lang, { allowQueue: false });
    }
    return { queue: [], recent: [], lastPicked: '', rev: 0 };
  } catch {
    return { queue: [], recent: [], lastPicked: '', rev: 0 };
  }
}

export class SentenceDeck {
  constructor(storage = getPersistentStorage()) {
    this.storage = storage;
    this.decks = Object.fromEntries(
      Object.keys(SENTENCE_POOLS).map((lang) => [lang, restoreDeck(storage, lang)])
    );
  }

  _syncFromStorage(lang) {
    if (!this.storage) return;
    try {
      const raw = this.storage.getItem(storageKey(lang));
      if (!raw) return;
      const remote = parseStoredDeck(raw, lang, { allowQueue: true });
      if (remote.rev > this.decks[lang].rev) {
        this.decks[lang] = remote;
      }
    } catch {
      // Ignore storage read errors and continue with in-memory state.
    }
  }

  next(lang = 'ko', { exclude = [] } = {}) {
    const key = lang === 'en' ? 'en' : 'ko';
    this._syncFromStorage(key);
    const deck = this.decks[key];
    if (deck.queue.length === 0) {
      deck.queue = buildCycleQueue(key, deck.recent);
    }

    const poolSet = POOL_SETS[key];
    const excludeSet = new Set(
      Array.isArray(exclude) ? exclude.filter((s) => typeof s === 'string' && poolSet.has(s)) : []
    );
    let pickIdx = excludeSet.size > 0 ? deck.queue.findIndex((s) => !excludeSet.has(s)) : 0;
    if (pickIdx < 0) pickIdx = 0;

    const [picked] = deck.queue.splice(pickIdx, 1);
    deck.lastPicked = picked;
    deck.recent = sanitizeRecent(deck.recent, picked, poolSet, cooldownSize(key));
    deck.rev = (deck.rev || 0) + 1;

    try {
      this.storage?.setItem(
        storageKey(key),
        JSON.stringify({
          queue: deck.queue,
          recent: deck.recent,
          lastPicked: deck.lastPicked,
          rev: deck.rev,
        })
      );
    } catch {
      // Keep dealing from memory when browser storage is unavailable.
    }
    return picked;
  }
}
