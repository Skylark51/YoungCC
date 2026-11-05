# Lion Gaussian 16 계산 레퍼런스

Lion 서버의 `/home/skylark/test`에서 가져온 Gaussian 입력(`.inp` 31개)과 출력(`.out` 17개)이다. MalonicAcid 및 n-butane 계산의 원본 파일명과 재시작 검증 하위 폴더 구조를 유지했다.

출력 17개 중 13개는 Gaussian 16 정상 종료 기록이 있다. 나머지 4개는 루트와 재시작 검증 폴더의 `*scan.out` 3개 및 `MalonicAcid-restart-check.out`이며, 비정상 종료 기록을 포함한다.

일부 입력은 선행 계산의 체크포인트를 참조한다. 이 폴더에는 체크포인트 파일이 없으므로 해당 입력을 재실행하려면 참조하는 체크포인트가 필요하다.
