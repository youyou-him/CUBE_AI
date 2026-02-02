import pandas as pd
from sqlalchemy import create_engine
import sys
import os
import glob

# --- 1. 사용자 설정 ---
CSV_FOLDER_PATH = 'C:/CUBE/CUBE/csv_data'
DB_PASSWORD = '0000'  # 실제 MySQL 비밀번호로 변경해주세요.

# --- 2. 데이터베이스 연결 정보 ---
DB_USER = 'cube_user'
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'cube_crm'


# --- 3. 데이터베이스 연결 및 데이터 삽입 ---
def import_all_csv_to_db():
    try:
        engine_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(engine_url)
        print("✅ 데이터베이스 연결에 성공했습니다.")

        csv_files = glob.glob(os.path.join(CSV_FOLDER_PATH, '*.csv'))
        if not csv_files:
            print(f"❌ '{CSV_FOLDER_PATH}' 폴더에서 CSV 파일을 찾을 수 없습니다.")
            return

        # (★★★★★ 추가된 부분: 번역 파일 먼저 로드 ★★★★★)
        translation_path = os.path.join(CSV_FOLDER_PATH, 'product_category_name_translation.csv')
        if os.path.exists(translation_path):
            translation_df = pd.read_csv(translation_path)
            print("✅ 카테고리 번역 파일을 로드했습니다.")
        else:
            translation_df = None
            print("⚠️ 카테고리 번역 파일을 찾을 수 없어, 영어 이름 없이 진행합니다.")

        print(f"\n총 {len(csv_files)}개의 CSV 파일 데이터 삽입을 시작합니다...")

        for file_path in csv_files:
            try:
                file_name = os.path.basename(file_path)
                table_name = file_name.replace('_dataset', '').replace('.csv', '')

                print(f"\n---\n🔄 작업 시작: '{file_name}' -> '{table_name}' 테이블")
                df = pd.read_csv(file_path)

                # (★★★★★ olist_products 파일 처리 시 번역 데이터 병합 ★★★★★)
                if file_name == 'olist_products_dataset.csv' and translation_df is not None:
                    df = pd.merge(df, translation_df, on='product_category_name', how='left')
                    print("  - 카테고리 영어 이름 데이터를 병합했습니다.")

                df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                df.to_sql(name=table_name, con=engine, if_exists='replace', index=False, chunksize=1000)
                print(f"  - ✅ '{table_name}' 테이블에 {len(df)}개 행 삽입 완료!")

            except Exception as e:
                print(f"  - ❌ '{file_name}' 파일 처리 중 오류 발생: {e}")
                continue

        print("\n🎉 모든 CSV 파일의 데이터베이스 삽입 작업이 완료되었습니다!")
    except Exception as e:
        print(f"\n[치명적 오류] 데이터베이스 연결 또는 중요 작업 중 문제가 발생했습니다: {e}")


if __name__ == '__main__':
    import_all_csv_to_db()

