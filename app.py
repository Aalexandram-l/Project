import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="ML Классификатор", layout="wide")

import streamlit as st
import joblib
from catboost import CatBoostClassifier

@st.cache_resource
def load_model(path, model_type="joblib"):
    """Универсальная загрузка моделей разных типов"""
    try:
        if model_type == "catboost":
            model = CatBoostClassifier()
            model.load_model(path)
            return model
        elif model_type == "tensorflow":
            return tf.keras.models.load_model(path)
        else:  # joblib (sklearn модели)
            return joblib.load(path)
    except Exception as e:
        st.error(f"Ошибка загрузки {path}: {e}")
        return None

# Правильные пути к файлам (с учётом реальных расширений)
models = {
    "ML1: SVM": {"path": "models/model_ml1.joblib", "type": "joblib"},
    "ML2: GradientBoosting (бустинг)": {"path": "models/model_ml2.joblib", "type": "joblib"},
    "ML3: CatBoost": {"path": "models/model_ml3.cbm", "type": "catboost"},
    "ML4: Random Forest (бэггинг)": {"path": "models/model_ml4.joblib", "type": "joblib"},
    "ML5: Stacking": {"path": "models/model_ml5.joblib", "type": "joblib"},
    "ML6: Нейронная сеть (MLP)": {"path": "models/model_ml6.joblib", "type": "joblib"}
}

# Загружаем все модели
loaded_models = {}
for name, model_info in models.items():
    loaded_models[name] = load_model(model_info["path"], model_info["type"])
    if loaded_models[name] is not None:
        st.success(f"✅ {name} загружена")
    else:
        st.warning(f"❌ {name} не загружена")

def page_developer():
    st.title("📄 О разработчике")
    col1, col2 = st.columns([1, 3])
    with col1:
        # Замените на путь к вашему фото
        st.image("foto.jpg", caption="Фото студента")
    with col2:
        st.markdown("""
        **ФИО:** Мирошникова Александра Викторовна 
        **Группа:** ФИТ-242
        **Тема РГР:** Разработка Web-приложения для инференса моделей ML (классификация)
        """)

def page_dataset():
    st.title("📊 Информация о наборе данных")
    st.markdown("""
    **Предметная область:** Киберспорт (CS:GO) – предсказание установки бомбы командой террористов.
    
    **Цель задачи:** Определить по текущему состоянию игры, установит ли команда террористов бомбу в текущем раунде.
    
    **Признаки:**
    - `time_left` – оставшееся время в раунде (секунды)
    - `ct_score` – текущий счёт команды спецназа (количество выигранных раундов)
    - `t_score` – текущий счёт команды террористов (количество выигранных раундов)
    - `map` – название игровой карты (категориальный признак)
    - `ct_health` – суммарное здоровье команды спецназа (HP)
    - `ct_armor` – суммарная броня команды спецназа
    - `t_armor` – суммарная броня команды террористов
    - `ct_money` – общая сумма денег команды спецназа ($)
    - `t_money` – общая сумма денег команды террористов ($)
    - `ct_helmets` – количество шлемов у команды спецназа
    - `t_helmets` – количество шлемов у команды террористов
    - `ct_defuse_kits` – количество наборов для разминирования
    - `ct_players_alive` – количество живых игроков в команде спецназа
    - `t_players_alive` – количество живых игроков в команде террористов
    
    **Целевая переменная:** `bomb_planted` – была ли установлена бомба в раунде (1 – да, 0 – нет)
    
    **Предобработка данных:**
    - Удаление дубликатов строк
    - Преобразование `bomb_planted` из True/False в 0/1
    - Кодирование категориального признака `map` (Label Encoding)
    - Масштабирование числовых признаков через StandardScaler
    - Разделение на обучающую и тестовую выборки (80% / 20%)
    
    **Особенности датасета:**
    - Несбалансированность классов (установка бомбы происходит не в каждом раунде)
    - Временная зависимость между последовательными раундами
    - Различные игровые карты имеют разную тактику установки бомбы
    """)

def page_visualizations():
    st.title("📈 Визуализации данных")
    
    # Загрузка реального датасета (укажите путь к вашему файлу)
    # Если файл в той же папке что и app.py:
    df = pd.read_csv('csgo_task_treated.csv')  # замените на имя вашего файла
    
    # Удаление дубликатов
    df = df.drop_duplicates()
    
    # Преобразование bomb_planted в 0/1 (если ещё не преобразовано)
    if df['bomb_planted'].dtype == 'bool':
        df['bomb_planted'] = df['bomb_planted'].astype(int)
    
    st.markdown("**Всего записей:** " + str(len(df)))
    
    # 1. Гистограмма оставшегося времени в раунде
    st.subheader("Распределение оставшегося времени в раунде")
    fig1, ax1 = plt.subplots()
    ax1.hist(df['time_left_sec'], bins=30, edgecolor='black', color='skyblue')
    ax1.set_xlabel("Оставшееся время (секунды)")
    ax1.set_ylabel("Частота")
    ax1.axvline(x=60, color='red', linestyle='--', label='60 секунд')
    ax1.legend()
    st.pyplot(fig1)
    
    # 2. Boxplot денег террористов по целевой переменной
    st.subheader("Деньги террористов в зависимости от установки бомбы")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='bomb_planted', y='t_money', data=df, ax=ax2)
    ax2.set_xlabel("Установка бомбы (0=нет, 1=да)")
    ax2.set_ylabel("Деньги террористов ($)")
    st.pyplot(fig2)
    
    # 3. Корреляционная матрица (только числовые признаки)
    st.subheader("Корреляционная матрица признаков")
    # Выбираем только числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Убираем bomb_planted из корреляции с самим собой
    corr_matrix = df[numeric_cols].corr()
    
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, ax=ax3, cbar_kws={"shrink": 0.8})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig3)
    
    # 4. Количество живых игроков в зависимости от установки бомбы
    st.subheader("Влияние живых игроков на установку бомбы")
    fig4, ax4 = plt.subplots()
    
    # Группировка по количеству живых террористов
    t_alive_means = df.groupby('t_players_alive')['bomb_planted'].mean()
    x_labels = t_alive_means.index.tolist()
    y_values = t_alive_means.values.tolist()
    
    ax4.bar(x_labels, y_values, color='orange', edgecolor='black')
    ax4.set_xlabel("Количество живых террористов")
    ax4.set_ylabel("Доля раундов с установкой бомбы")
    ax4.set_ylim(0, 1)
    ax4.set_xticks(range(1, 6))
    st.pyplot(fig4)
    
    # 5. Дополнительная визуализация: здоровье CT vs установка бомбы
    st.subheader("Здоровье спецназа в зависимости от установки бомбы")
    fig5, ax5 = plt.subplots()
    sns.violinplot(x='bomb_planted', y='ct_health', data=df, ax=ax5)
    ax5.set_xlabel("Установка бомбы (0=нет, 1=да)")
    ax5.set_ylabel("Суммарное здоровье спецназа (HP)")
    st.pyplot(fig5)

def page_inference():
    st.title("🔮 Предсказание установки бомбы (bomb_planted)")
    
    # Выбор модели
    model_name = st.selectbox("Выберите модель", list(loaded_models.keys()))
    model = loaded_models[model_name]
    
    if model is None:
        st.error("Модель не загружена. Проверьте путь к файлу.")
        return
    
    # Специальная обработка для CatBoost
    if "CatBoost" in model_name:
        expected_features = 16  # явно указываем количество признаков
    else:
        expected_features = model.n_features_in_ if hasattr(model, "n_features_in_") and model.n_features_in_ > 0 else 16
    
    # Показываем информацию в боковой панели
    st.sidebar.info(f"Модель **{model_name}** ожидает **{expected_features}** признаков")
    
    # Способ ввода данных
    input_method = st.radio("Способ ввода данных", ["Ручной ввод", "Загрузка CSV"])
    
    # ----- Ручной ввод -----
    if input_method == "Ручной ввод":
        st.subheader("Введите текущее состояние игры")
        
        if expected_features == 10:
            st.info("⚙️ Эта модель использует 10 признаков (здоровье, деньги, броня, время)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**❤️ Здоровье и броня**")
                ct_health = st.number_input("Здоровье CT", min_value=0, max_value=500, value=400)
                t_health = st.number_input("Здоровье T", min_value=0, max_value=500, value=400)
                ct_armor = st.number_input("Броня CT", min_value=0, max_value=500, value=200)
                
                st.markdown("**💰 Экономика**")
                ct_money = st.number_input("Деньги CT ($)", min_value=0, max_value=50000, value=5000, step=500)
                ct_helmets = st.number_input("Шлемы CT", min_value=0, max_value=5, value=0)
            
            with col2:
                st.markdown("**🔧 Снаряжение**")
                ct_defuse_kits = st.number_input("Наборы для разминирования", min_value=0, max_value=5, value=0)
                
                st.markdown("**👥 Живые игроки**")
                ct_players_alive = st.number_input("Живые CT", min_value=0, max_value=5, value=5)
                t_players_alive = st.number_input("Живые T", min_value=0, max_value=5, value=5)
                
                st.markdown("**⏱️ Время**")
                time_left_sec = st.number_input("Оставшееся время (сек)", min_value=0.0, max_value=180.0, value=90.0, step=1.0)
            
            time_left_ms = time_left_sec * 1000
            
            if st.button("🔮 Предсказать"):
                # 10 признаков для MLPClassifier
                input_data = np.array([[ct_health, t_health, ct_armor, ct_money, ct_helmets,
                                        ct_defuse_kits, ct_players_alive, t_players_alive,
                                        time_left_sec, time_left_ms]])
                
                prediction = model.predict(input_data)[0]
                
                # Получаем вероятности, если возможно
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_data)[0]
                    prob_bomb = proba[1]
                    st.write(f"**Вероятность установки бомбы:** {prob_bomb:.2%}")
                
                st.markdown("---")
                if prediction == 1:
                    st.success("💣 **БОМБА БУДЕТ УСТАНОВЛЕНА!**")
                else:
                    st.info("🔫 **БОМБА НЕ БУДЕТ УСТАНОВЛЕНА**")
        
        else:  # expected_features == 16
            st.info("⚙️ Эта модель использует 16 признаков (полный набор данных)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**⏱️ Время и счёт**")
                time_left_sec = st.number_input("Оставшееся время (сек)", min_value=0.0, max_value=180.0, value=90.0, step=1.0)
                ct_score = st.number_input("Счёт CT", min_value=0, max_value=30, value=0)
                t_score = st.number_input("Счёт T", min_value=0, max_value=30, value=0)
                map_binary = st.selectbox("map_binary", [0, 1], format_func=lambda x: "Карта 0" if x == 0 else "Карта 1")
            
            with col2:
                st.markdown("**❤️ Здоровье и броня**")
                ct_health = st.number_input("Здоровье CT", min_value=0, max_value=500, value=400)
                t_health = st.number_input("Здоровье T", min_value=0, max_value=500, value=400)
                ct_armor = st.number_input("Броня CT", min_value=0, max_value=500, value=200)
                t_armor = st.number_input("Броня T", min_value=0, max_value=500, value=200)
            
            with col3:
                st.markdown("**💰 Экономика**")
                ct_money = st.number_input("Деньги CT ($)", min_value=0, max_value=50000, value=5000, step=500)
                t_money = st.number_input("Деньги T ($)", min_value=0, max_value=50000, value=5000, step=500)
                ct_helmets = st.number_input("Шлемы CT", min_value=0, max_value=5, value=0)
                t_helmets = st.number_input("Шлемы T", min_value=0, max_value=5, value=0)
                ct_defuse_kits = st.number_input("Наборы для разминирования", min_value=0, max_value=5, value=0)
                ct_players_alive = st.number_input("Живые CT", min_value=0, max_value=5, value=5)
                t_players_alive = st.number_input("Живые T", min_value=0, max_value=5, value=5)
            
            time_left_ms = time_left_sec * 1000
            
            if st.button("🔮 Предсказать"):
                # 16 признаков
                input_data = np.array([[ct_score, t_score, ct_health, t_health, 
                                        ct_armor, t_armor, ct_money, t_money, 
                                        ct_helmets, t_helmets, ct_defuse_kits,
                                        ct_players_alive, t_players_alive, 
                                        map_binary, time_left_sec, time_left_ms]])
                
                prediction = model.predict(input_data)[0]
                
                # Получаем вероятности, если возможно
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_data)[0]
                    prob_bomb = proba[1]
                    st.write(f"**Вероятность установки бомбы:** {prob_bomb:.2%}")
                
                st.markdown("---")
                if prediction == 1:
                    st.success("💣 **БОМБА БУДЕТ УСТАНОВЛЕНА!**")
                else:
                    st.info("🔫 **БОМБА НЕ БУДЕТ УСТАНОВЛЕНА**")
    
    # ----- Загрузка CSV -----
    else:
        st.subheader("Загрузите файл с данными (CSV)")
        
        # Определяем ожидаемые колонки в зависимости от модели
        if expected_features == 10:
            expected_cols = ['ct_health', 't_health', 'ct_armor', 'ct_money', 'ct_helmets',
                            'ct_defuse_kits', 'ct_players_alive', 't_players_alive', 
                            'time_left_sec', 'time_left_ms']
            st.info(f"Модель ожидает {len(expected_cols)} колонок: {', '.join(expected_cols)}")
        else:
            expected_cols = ['ct_score', 't_score', 'ct_health', 't_health', 
                            'ct_armor', 't_armor', 'ct_money', 't_money', 
                            'ct_helmets', 't_helmets', 'ct_defuse_kits',
                            'ct_players_alive', 't_players_alive', 'map_binary', 
                            'time_left_sec', 'time_left_ms']
            st.info(f"Модель ожидает {len(expected_cols)} колонок")
        
        uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Предпросмотр загруженных данных:")
            st.dataframe(df.head(10))
            
            # Автоматически создаём time_left_ms, если есть time_left_sec
            if 'time_left_ms' not in df.columns and 'time_left_sec' in df.columns:
                df['time_left_ms'] = df['time_left_sec'] * 1000
                st.info("Добавлена колонка 'time_left_ms' на основе 'time_left_sec'")
            
            # Проверка наличия всех ожидаемых колонок
            missing_cols = set(expected_cols) - set(df.columns)
            if missing_cols:
                st.error(f"Файл не содержит колонки: {missing_cols}")
                return
            
            if st.button("Предсказать для всех строк"):
                X = df[expected_cols].values
                predictions = model.predict(X)
                
                df['prediction'] = predictions
                df['prediction_label'] = df['prediction'].map({1: "Бомба будет", 0: "Бомбы не будет"})
                
                # Добавляем вероятности, если возможно
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)
                    df['probability_bomb'] = proba[:, 1]
                    display_cols = expected_cols[:3] + ['prediction_label', 'probability_bomb']
                else:
                    display_cols = expected_cols[:3] + ['prediction_label']
                
                st.subheader("Результаты предсказаний:")
                st.dataframe(df[display_cols].head(20))
                
                # Статистика
                st.subheader("📊 Статистика предсказаний")
                col_a, col_b = st.columns(2)
                with col_a:
                    bomb_count = df['prediction'].sum()
                    st.metric("Количество раундов с установкой бомбы", f"{bomb_count} / {len(df)}")
                with col_b:
                    bomb_percent = bomb_count / len(df) * 100
                    st.metric("Доля установки бомбы", f"{bomb_percent:.1f}%")
                
                # Кнопка скачивания результата
                csv_output = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Скачать результаты CSV",
                    csv_output,
                    "bomb_planted_predictions.csv",
                    "text/csv"
                )
 

st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти на страницу", 
                        ["О разработчике", "О датасете", "Визуализации", "Инференс"])

if page == "О разработчике":
    page_developer()
elif page == "О датасете":
    page_dataset()
elif page == "Визуализации":
    page_visualizations()
elif page == "Инференс":
    page_inference()
