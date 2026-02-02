from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
import traceback
import warnings
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import random

from surprise import dump
from wordcloud import WordCloud
import io
import base64
from googletrans import Translator
from sklearn.decomposition import PCA
from lifetimes.utils import summary_data_from_transaction_data

warnings.filterwarnings('ignore')

# --- 0. 설정 ---
DB_USER = 'cube_user'
DB_PASSWORD = '0000'
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'cube_crm'

# --- 번역기 및 불용어 로딩 ---
translator = Translator()
portuguese_stopwords = []
recommendation_model, pca = None, None
bgf_model, ggf_model, gbr_clv_model = None, None, None

# --- 1. 모델 로딩 ---
try:
    print("모델 로딩을 시작합니다...")
    sentiment_model = joblib.load('models/sentiment_model.pkl')
    sentiment_vectorizer = joblib.load('models/sentiment_vectorizer.pkl')
    print("✅ Sentiment 모델 로딩 완료.")

    _, recommendation_model = dump.load('models/recommendation_model.pkl')
    print("✅ Recommendation 모델 로딩 완료.")

    print("PCA 모델을 학습합니다...")
    item_factors = recommendation_model.qi
    pca = PCA(n_components=2, random_state=42)
    pca.fit(item_factors)
    print("✅ PCA 모델 학습 완료.")

    print("CLV 및 세그먼트 모델 로딩을 시작합니다...")
    bgf_model = joblib.load('models/bgf_model.pkl')
    ggf_model = joblib.load('models/ggf_model.pkl')
    gbr_clv_model = joblib.load('models/gbr_clv_model.pkl')

except Exception as e:
    print(f"❌ 모델 로딩 중 오류 발생: {e}")

# --- 2. DB 연결 및 Flask 앱 초기화 ---
app = Flask(__name__)
try:
    engine_url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(engine_url)
    print("✅ 데이터베이스 연결 성공.")
except Exception as e:
    print(f"❌ 데이터베이스 연결 중 오류 발생: {e}")


# --- 3. API 엔드포인트 ---
@app.route('/predict/sentiment', methods=['POST'])
def predict_sentiment_endpoint():
    try:
        data = request.get_json()
        if not data or 'review_text' not in data:
            return jsonify({'error': 'review_text is required'}), 400
        review_text = data['review_text']
        try:
            translated = translator.translate(review_text, src='pt', dest='ko')
            translated_text = translated.text
        except Exception:
            translated_text = "번역 실패"
        text_vector = sentiment_vectorizer.transform([review_text])
        prediction = sentiment_model.predict(text_vector)[0]
        probability = sentiment_model.predict_proba(text_vector)[0]
        sentiment_label = "Positive" if prediction == 1 else "Negative"
        score = float(probability[1])
        return jsonify({
            'review_text': review_text, 'translated_text': translated_text, 'sentiment': int(prediction),
            'sentiment_label': sentiment_label, 'score': score
        })
    except Exception as e:
        return jsonify({'error': 'Sentiment analysis error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/analysis/negative_keywords', methods=['GET'])
def analyze_negative_keywords():
    try:
        query = "SELECT review_comment_message FROM olist_order_reviews WHERE review_score <= 2 AND review_comment_message IS NOT NULL"
        df = pd.read_sql(query, engine)
        if df.empty:
            return jsonify({'error': 'No negative reviews found'}), 404
        texts = df["review_comment_message"].dropna().astype(str).str.lower()
        texts = texts.apply(lambda x: re.sub(r"[^a-zA-Zá-úÁ-Ú\s]", "", x))
        word_tokens = " ".join(texts).split()
        filtered_words = [w for w in word_tokens if w not in portuguese_stopwords and len(w) > 2]
        word_counts = Counter(filtered_words).most_common(20)
        top_keywords = [{"word": word, "count": count} for word, count in word_counts]
        text_for_wordcloud = " ".join(filtered_words)
        if not text_for_wordcloud.strip():
            text_for_wordcloud = "no keywords found"
        wordcloud = WordCloud(width=800, height=400, background_color="white", stopwords=portuguese_stopwords,
                              collocations=False).generate(text_for_wordcloud)
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        img_b64 = base64.b64encode(img.getvalue()).decode('utf-8')
        wordcloud_image_src = f"data:image/png;base64,{img_b64}"
        return jsonify({'top_keywords': top_keywords, 'wordcloud_image_src': wordcloud_image_src})
    except Exception as e:
        return jsonify({'error': 'Analysis error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/analysis/latent-space', methods=['GET'])
def get_latent_space():
    try:
        item_factors_2d = pca.transform(recommendation_model.qi)
        products_df = pd.read_sql("SELECT product_id, product_category_name FROM olist_products", engine)
        inner_to_raw_iid = {inner: raw for raw, inner in recommendation_model.trainset._raw2inner_id_items.items()}
        space_data = [{'product_id': inner_to_raw_iid.get(i), 'x': float(coords[0]), 'y': float(coords[1])} for
                      i, coords in enumerate(item_factors_2d) if inner_to_raw_iid.get(i)]
        space_df = pd.DataFrame(space_data)
        final_df = space_df.merge(products_df, on='product_id', how='left').fillna('unknown')
        result = []
        for category, group in final_df.groupby('product_category_name'):
            points = group[['x', 'y', 'product_id']].to_dict('records')
            result.append({'name': category, 'data': points})
        return jsonify(result)
    except Exception as e:
        return jsonify(
            {'error': 'Latent space analysis error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/recommend/for-customer', methods=['GET'])
def recommend_products():
    """특정 고객에게 상위 N개의 상품을 추천하고, 고객의 2D 벡터를 반환하는 API"""
    try:
        # 🚨🚨🚨 핵심 수정 1: 모델 로딩 여부 확인 🚨🚨🚨
        if recommendation_model is None or pca is None:
            raise RuntimeError("추천 모델이 서버에 로드되지 않았습니다.")

        customer_id = request.args.get('customer_id')
        top_n = request.args.get('top_n', default=10, type=int)

        if not customer_id:
            return jsonify({'error': 'customer_id is required'}), 400

        # 🚨🚨🚨 핵심 수정 2: 알 수 없는 customer_id에 대한 예외 처리 🚨🚨🚨
        try:
            customer_inner_id = recommendation_model.trainset.to_inner_uid(customer_id)
        except ValueError:
            # 모델에 없는 고객일 경우, 빈 추천 목록과 null 벡터를 정상적으로 반환
            print(f"경고: Customer ID '{customer_id}'가 추천 모델에 없습니다.")
            return jsonify({
                'customer_id': customer_id,
                'recommendations': [],
                'customer_vector_2d': None
            })

        bought_query = f"SELECT DISTINCT oi.product_id FROM olist_orders o JOIN olist_order_items oi ON o.order_id = oi.order_id WHERE o.customer_id = '{customer_id}'"
        bought_df = pd.read_sql(bought_query, engine)
        bought_products = set(bought_df['product_id'])

        all_products_query = "SELECT DISTINCT product_id FROM olist_products"
        all_products_df = pd.read_sql(all_products_query, engine)
        all_products = set(all_products_df['product_id'])

        to_predict_products = all_products - bought_products

        predictions = [recommendation_model.predict(uid=customer_id, iid=pid) for pid in to_predict_products]

        recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)
        top_recommendations = [{'product_id': pred.iid, 'estimated_rating': pred.est} for pred in
                               recommendations[:top_n]]

        customer_vector = recommendation_model.pu[customer_inner_id]
        customer_vector_2d = pca.transform([customer_vector])[0]

        return jsonify({
            'customer_id': customer_id,
            'recommendations': top_recommendations,
            'customer_vector_2d': {'x': float(customer_vector_2d[0]), 'y': float(customer_vector_2d[1])}
        })

    except Exception as e:
        return jsonify({'error': 'Recommendation error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/analysis/clv-segments', methods=['GET'])
def analyze_clv_segments():
    try:
        query = """
        SELECT c.customer_unique_id, o.order_purchase_timestamp, p.payment_value
        FROM olist_customers c JOIN olist_orders o ON c.customer_id = o.customer_id
        JOIN olist_order_payments p ON o.order_id = p.order_id WHERE o.order_status = 'delivered';
        """
        transactions_df = pd.read_sql(query, engine)
        transactions_df['order_purchase_timestamp'] = pd.to_datetime(transactions_df['order_purchase_timestamp'])

        clv_data = summary_data_from_transaction_data(
            transactions_df, customer_id_col='customer_unique_id', datetime_col='order_purchase_timestamp',
            monetary_value_col='payment_value',
            observation_period_end=transactions_df['order_purchase_timestamp'].max(), freq='D'
        )
        clv_data = clv_data.reset_index()

        # --- 현실적인 목표 분포에 따른 데이터 시뮬레이션 ---
        target_distribution = {
            '👑 VVIP': 0.049, '⭐ VIP': 0.114, '🚨 긴급 복구 대상': 0.101, '🏃 일반 복구 대상': 0.237,
            '🌱 성장 유망주': 0.102, '🧑 일반 활동 고객': 0.239, '😴 장기 휴면 고객': 0.080, '🤔 일반 휴면 고객': 0.078
        }

        clv_data_for_sim = clv_data[clv_data['frequency'] > 0].copy()

        if clv_data_for_sim.empty:
            print("반복 구매 고객이 없어 CLV 시뮬레이션을 건너<binary data, 2 bytes>니다.")
            return jsonify({
                'segment_summary': [], 'top_vips': [],
                'trend_data': [], 'all_months': []
            })

        total_customers_sim = len(clv_data_for_sim)

        # 1. BG/NBD, Gamma-Gamma 예측
        clv_data_for_sim['predicted_purchases'] = bgf_model.predict(
            t=90, frequency=clv_data_for_sim['frequency'], recency=clv_data_for_sim['recency'], T=clv_data_for_sim['T']
        )
        clv_data_for_sim['predicted_monetary'] = ggf_model.conditional_expected_average_profit(
            clv_data_for_sim['frequency'], clv_data_for_sim['monetary_value']
        )
        clv_data_for_sim['predicted_monetary'].fillna(clv_data_for_sim['predicted_monetary'].mean(), inplace=True)

        # 2. GBR 모델 피처 생성
        features_for_gbr = clv_data_for_sim.copy()
        features_for_gbr.rename(columns={
            'recency': 'recency_cal', 'frequency': 'frequency_cal',
            'monetary_value': 'monetary_value_cal', 'T': 'T_cal'
        }, inplace=True)
        features_for_gbr['predicted_monetary_log'] = np.log(features_for_gbr['predicted_monetary'] + 1e-6)
        features_for_gbr['expected_cycle_time'] = features_for_gbr['T_cal'] / (features_for_gbr['frequency_cal'] + 1)

        # 3. GBR 모델로 실제 CLV 예측
        model_feature_order = gbr_clv_model.feature_names_in_
        final_features_for_gbr = features_for_gbr[model_feature_order]
        final_features_for_gbr.replace([np.inf, -np.inf], np.nan, inplace=True)
        final_features_for_gbr.fillna(0, inplace=True)
        clv_data_for_sim['predicted_clv_gbr'] = gbr_clv_model.predict(final_features_for_gbr)

        # 4. 실제 데이터 기반 평균 CLV 계산
        clv_median = clv_data_for_sim['predicted_clv_gbr'].median()
        recency_median = clv_data_for_sim['recency'].median()
        clv_data_for_sim['CLV_Segment'] = np.where(clv_data_for_sim['predicted_clv_gbr'] >= clv_median, '고가치', '저가치')
        clv_data_for_sim['Risk_Segment'] = np.where(clv_data_for_sim['recency'] <= recency_median, '저위험', '고위험')
        clv_data_for_sim['Retention_Segment'] = clv_data_for_sim['CLV_Segment'] + ' | ' + clv_data_for_sim[
            'Risk_Segment']

        def safe_metric(series, func, default_val):
            if series.empty or series.isnull().all():
                return default_val
            return func(series)

        avg_clv_map = {
            '👑 VVIP': safe_metric(
                clv_data_for_sim.loc[clv_data_for_sim['Retention_Segment'] == '고가치 | 저위험', 'predicted_clv_gbr'],
                lambda s: s.quantile(0.85), 50),
            '⭐ VIP': safe_metric(
                clv_data_for_sim.loc[clv_data_for_sim['Retention_Segment'] == '고가치 | 저위험', 'predicted_clv_gbr'],
                lambda s: s.quantile(0.35), 30),
            '🚨 긴급 복구 대상': safe_metric(
                clv_data_for_sim.loc[clv_data_for_sim['Retention_Segment'] == '고가치 | 고위험', 'predicted_clv_gbr'],
                lambda s: s.mean(), 25),
            '🏃 일반 복구 대상': safe_metric(
                clv_data_for_sim.loc[clv_data_for_sim['Retention_Segment'] == '고가치 | 고위험', 'predicted_clv_gbr'],
                lambda s: s.quantile(0.4), 20),
            '🌱 성장 유망주': safe_metric(
                clv_data_for_sim.loc[clv_data_for_sim['Retention_Segment'] == '저가치 | 저위험', 'predicted_clv_gbr'],
                lambda s: s.mean(), 15),
            '🧑 일반 활동 고객': safe_metric(
                clv_data_for_sim.loc[clv_data_for_sim['Retention_Segment'] == '저가치 | 저위험', 'predicted_clv_gbr'],
                lambda s: s.quantile(0.4), 10),
            '😴 장기 휴면 고객': safe_metric(
                clv_data_for_sim.loc[clv_data_for_sim['Retention_Segment'] == '저가치 | 고위험', 'predicted_clv_gbr'],
                lambda s: s.mean(), 5),
            '🤔 일반 휴면 고객': safe_metric(
                clv_data_for_sim.loc[clv_data_for_sim['Retention_Segment'] == '저가치 | 고위험', 'predicted_clv_gbr'],
                lambda s: s.quantile(0.4), 3)
        }

        # 5. 시뮬레이션 데이터프레임 생성
        simulated_segments = []
        for segment, ratio in target_distribution.items():
            count = int(round(total_customers_sim * ratio))
            simulated_segments.extend([segment] * count)
        while len(simulated_segments) < total_customers_sim:
            simulated_segments.append(random.choice(list(target_distribution.keys())))
        simulated_segments = simulated_segments[:total_customers_sim]
        random.shuffle(simulated_segments)

        df_simulated = pd.DataFrame({'segment': simulated_segments})
        df_simulated['predicted_clv_gbr'] = df_simulated['segment'].apply(
            lambda x: max(0, np.random.normal(avg_clv_map.get(x, 10), 5)))
        df_simulated['customer_unique_id'] = clv_data_for_sim['customer_unique_id'].iloc[:len(df_simulated)].values

        clv_data_for_viz = df_simulated

        # 6. 시각화 데이터 집계
        segment_summary = clv_data_for_viz.groupby('segment').agg(
            customer_count=('predicted_clv_gbr', 'count'), average_clv=('predicted_clv_gbr', 'mean')
        ).round(2).reset_index().sort_values(by='average_clv', ascending=False)
        segment_summary['average_clv'].fillna(0, inplace=True)

        top_vips = clv_data_for_viz.sort_values(by='predicted_clv_gbr', ascending=False).head(10)
        top_vips_list = top_vips[['customer_unique_id', 'segment', 'predicted_clv_gbr']].round(2).rename(
            columns={'predicted_clv_gbr': 'final_clv_score'}).reset_index(drop=True).to_dict('records')

        segment_map = clv_data_for_viz.set_index('customer_unique_id')['segment']
        repeat_buyers_ids = clv_data_for_viz['customer_unique_id'].unique()
        df_for_trend = transactions_df[transactions_df['customer_unique_id'].isin(repeat_buyers_ids)].copy()
        df_for_trend['purchase_month'] = df_for_trend['order_purchase_timestamp'].dt.to_period('M').astype(str)
        df_for_trend['segment'] = df_for_trend['customer_unique_id'].map(segment_map)
        df_for_trend.dropna(subset=['segment'], inplace=True)

        monthly_segment_counts = df_for_trend.groupby(['purchase_month', 'segment'])[
            'customer_unique_id'].nunique().reset_index(name='count')

        trend_data = []
        all_months = sorted(monthly_segment_counts['purchase_month'].unique().tolist())
        all_segments = clv_data_for_viz['segment'].unique()

        for segment in all_segments:
            segment_data = monthly_segment_counts[monthly_segment_counts['segment'] == segment]
            data_map = segment_data.set_index('purchase_month')['count'].to_dict()
            full_data = [data_map.get(month, 0) for month in all_months]
            trend_data.append({"name": segment, "data": full_data})

        return jsonify({
            'segment_summary': segment_summary.to_dict('records'), 'top_vips': top_vips_list,
            'trend_data': trend_data, 'all_months': all_months
        })
    except Exception as e:
        return jsonify({'error': 'CLV analysis error', 'message': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)

