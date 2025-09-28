import streamlit as st
import pandas as pd
from genetic_algorithm import run_genetic_algorithm
from sklearn.feature_selection import SelectKBest, chi2
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data.dropna(axis=1, how='all', inplace=True)
    return data

def get_model_accuracy(X, y, features):
    if not features:
        return 0
    X_subset = X[features]
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

st.set_page_config(layout="wide")

st.title("مشروع اختيار الميزات باستخدام الخوارزميات الجينية")

st.sidebar.header("تحميل البيانات والإعدادات")
uploaded_file = st.sidebar.file_uploader("ارفع ملف بيانات (CSV)", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)

    st.subheader("عينة من البيانات")
    st.dataframe(data.head())
    
    if 'id' in data.columns:
        data = data.drop(columns=['id'])
        st.caption("تم حذف عمود 'id' غير الضروري.")

    if len(data.columns) > 1:
        target_column = st.sidebar.selectbox("اختر المتغير الهدف (Target)", data.columns, index=0)
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        if y.dtype == 'object':
            unique_values = y.unique()
            if len(unique_values) == 2:
                st.caption(f"تم تحويل المتغير الهدف من قيم نصية {unique_values} إلى (0 و 1).")
                y = y.map({unique_values[0]: 0, unique_values[1]: 1})

        st.sidebar.subheader("معاملات الخوارزمية الجينية")
        population_size = st.sidebar.slider("حجم السكان (Population Size)", 10, 200, 50)
        num_generations = st.sidebar.slider("عدد الأجيال (Number of Generations)", 10, 100, 30)
        mutation_rate = st.sidebar.slider("معدل الطفرة (Mutation Rate)", 0.01, 1.0, 0.1, 0.01)

        non_numeric_cols = X.select_dtypes(include=['object', 'bool']).columns
        if len(non_numeric_cols) > 0:
            st.sidebar.subheader("التعامل مع الميزات غير الرقمية")
            st.sidebar.warning(f"تم العثور على ميزات غير رقمية: {', '.join(non_numeric_cols)}")
            non_numeric_action = st.sidebar.radio(
                "كيف تريد التعامل مع الميزات غير الرقمية؟",
                ("حذف الأعمدة", "تحويل باستخدام One-Hot Encoding")
            )

            if non_numeric_action == "حذف الأعمدة":
                X = X.drop(columns=non_numeric_cols)
                st.sidebar.info(f"تم حذف الأعمدة غير الرقمية: {', '.join(non_numeric_cols)}")
            elif non_numeric_action == "تحويل باستخدام One-Hot Encoding":
                X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)
                st.sidebar.info(f"تم تحويل الأعمدة غير الرقمية باستخدام One-Hot Encoding: {', '.join(non_numeric_cols)}")

        if not all(X.dtypes.apply(pd.api.types.is_numeric_dtype)):
            st.error("لا تزال هناك ميزات غير رقمية بعد المعالجة المسبقة. يرجى مراجعة البيانات.")
        else:
            if st.button("ابدأ عملية اختيار الميزات"):
                st.subheader("النتائج")
                
                with st.spinner("يتم الآن تشغيل الخوارزمية الجينية... قد يستغرق هذا بعض الوقت."):
                    progress_bar = st.progress(0)
                    progress_text = st.empty()

                    def ga_progress_callback(current_generation, total_generations, best_fitness):
                        progress_percent = (current_generation / total_generations)
                        progress_bar.progress(progress_percent)
                        progress_text.text(f"الجيل {current_generation}/{total_generations} | أفضل صلاحية: {best_fitness:.4f}")

                    ga_features, ga_fitness = run_genetic_algorithm(
                        X, y,
                        population_size=population_size,
                        generations=num_generations,
                        mutation_rate=mutation_rate,
                        progress_callback=ga_progress_callback
                    )
                
                st.success("انتهت الخوارزمية الجينية!")
                st.write(f"**أفضل صلاحية (Fitness) تم تحقيقها:** {ga_fitness:.4f}")
                st.write(f"**عدد الميزات المختارة:** {len(ga_features)}")
                st.write("**قائمة الميزات المثلى:**")
                st.json(ga_features)
                ga_accuracy = get_model_accuracy(X, y, ga_features)

                st.markdown("---")
                st.subheader("مقارنة مع طريقة إحصائية (SelectKBest with Chi-squared)")
                
                with st.spinner("يتم الآن تشغيل الطريقة الإحصائية..."):
                    k = len(ga_features)
                    if k > 0:
                        if (X < 0).any().any():
                            st.warning("تم العثور على قيم سالبة في البيانات، لا يمكن استخدام Chi-squared. سيتم تخطي المقارنة الإحصائية.")
                            stat_features = []
                            stat_accuracy = 0
                        else:
                            selector = SelectKBest(chi2, k=k)
                            selector.fit(X, y)
                            stat_features = list(X.columns[selector.get_support(indices=True)])
                            stat_accuracy = get_model_accuracy(X, y, stat_features)

                        st.write(f"**عدد الميزات المختارة:** {len(stat_features)}")
                        st.write("**قائمة الميزات المختارة:**")
                        st.json(stat_features)
                    else:
                        stat_features = []
                        stat_accuracy = 0


                st.markdown("---")
                st.subheader("مقارنة الأداء النهائية")
                
                results_df = pd.DataFrame({
                    'الطريقة': ['الخوارزمية الجينية', 'Chi-squared (SelectKBest)'],
                    'دقة النموذج': [ga_accuracy, stat_accuracy],
                    'عدد الميزات': [len(ga_features), len(stat_features)]
                })
                
                st.dataframe(results_df)

                fig = px.bar(results_df, x='الطريقة', y='دقة النموذج', color='الطريقة',
                             title="مقارنة دقة النموذج بين الطرق", text_auto=True)
                st.plotly_chart(fig)