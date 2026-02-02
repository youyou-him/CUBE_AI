-- 1. 'cube_crm' 이라는 새로운 데이터베이스(스키마)를 생성합니다.
CREATE SCHEMA `cube_crm` DEFAULT CHARACTER SET utf8mb4;

-- 2. 이 데이터베이스에 접속할 새로운 사용자를 생성합니다.
-- 'your_password' 부분은 사용할 실제 비밀번호로 변경해주세요.
CREATE USER 'cube_user'@'localhost' IDENTIFIED BY '0000';

-- 3. 생성한 사용자가 'cube_crm' 데이터베이스의 모든 테이블에 접근할 수 있도록 권한을 부여합니다.
GRANT ALL PRIVILEGES ON cube_crm.* TO 'cube_user'@'localhost';

-- 4. 변경된 권한을 시스템에 바로 적용합니다.
FLUSH PRIVILEGES;