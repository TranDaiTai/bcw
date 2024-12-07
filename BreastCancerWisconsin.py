# Bước 1: Import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

import graphviz  # Dùng để visualize decision tree (cần cài đặt graphviz system)
# Nếu cần: !pip install graphviz

####################################
# Bước 2: Đọc và chuẩn bị dữ liệu
####################################

# Ví dụ với dữ liệu Breast Cancer (có thể thay bằng link hoặc file CSV)
# Đọc dữ liệu Breast Cancer Wisconsin Diagnostic từ file CSV đã tải về
breast_cancer_data = pd.read_csv('wdbc.data')  # Chèn đường dẫn thực tế

# Đặt tên cho các cột dựa trên mô tả trong wdbc.names
columns = [
    'ID', 'Diagnosis',
    'Radius_Mean', 'Texture_Mean', 'Perimeter_Mean', 'Area_Mean', 'Smoothness_Mean', 'Compactness_Mean',
    'Concavity_Mean', 'Concave_Points_Mean', 'Symmetry_Mean', 'Fractal_Dimension_Mean',
    'Radius_SE', 'Texture_SE', 'Perimeter_SE', 'Area_SE', 'Smoothness_SE', 'Compactness_SE',
    'Concavity_SE', 'Concave_Points_SE', 'Symmetry_SE', 'Fractal_Dimension_SE',
    'Radius_Worst', 'Texture_Worst', 'Perimeter_Worst', 'Area_Worst', 'Smoothness_Worst', 'Compactness_Worst',
    'Concavity_Worst', 'Concave_Points_Worst', 'Symmetry_Worst', 'Fractal_Dimension_Worst'
]
# Hiển thị lại dữ liệu với tên cột
breast_cancer_data.columns = columns

#Cột ID không chứa thông tin quan trọng cho việc xây dựng mô hình nên bạn có thể loại bỏ: 
breast_cancer_data = breast_cancer_data.drop(columns=['ID'])    # Loại bỏ cột ID

# Kiểm tra dữ liệu
print(breast_cancer_data.head())
# print(breast_cancer_data.info())
# print(breast_cancer_data['Diagnosis'].value_counts())   # Kiểm tra phân phối nhãn


# Giả sử cột 'label' là nhãn, phần còn lại là features
features_bc = breast_cancer_data.drop('Diagnosis', axis=1)
labels_bc = breast_cancer_data['Diagnosis']

# Nếu label là chuỗi (M, B), chúng ta có thể encode thành 0/1
le = LabelEncoder()
labels_bc = le.fit_transform(labels_bc)  # M:1, B:0 chẳng hạn

# Tạo hàm tiện ích để chia dữ liệu theo nhiều tỉ lệ khác nhau và lưu lại
def stratified_split(X, y, train_size, random_state=42):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test

train_test_ratios = [0.4, 0.6, 0.8, 0.9]

splits_bc = {}  # Dictionary để lưu các tập dữ liệu
for ratio in train_test_ratios:
    X_train, X_test, y_train, y_test = stratified_split(features_bc, labels_bc, train_size=ratio)
    splits_bc[ratio] = (X_train, X_test, y_train, y_test)

# ####################################
# # Bước 3: Trực quan hóa phân bố lớp
# ####################################

def plot_class_distribution(y, title="Class Distribution"):
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(4,4))
    sns.barplot(x=unique, y=counts)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

# Vẽ phân bố lớp cho bộ dữ liệu gốc
plot_class_distribution(labels_bc, title="Original Data Class Distribution")

# Vẽ phân bố cho từng tập
for ratio in train_test_ratios:
    X_train, X_test, y_train, y_test = splits_bc[ratio]
    plot_class_distribution(y_train, title=f"Train Class Dist (ratio={int(ratio*100)}/{int((1-ratio)*100)})")
    plot_class_distribution(y_test, title=f"Test Class Dist (ratio={int(ratio*100)}/{int((1-ratio)*100)})")

# ####################################
# # Bước 4 & 5: Huấn luyện mô hình Decision Tree & Trực quan hóa cây
# ####################################

def train_and_visualize_decision_tree(X_train, y_train, max_depth=None):
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    return dt

def visualize_tree(dt, feature_names, class_names):
    dot_data = export_graphviz(
        dt, 
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True, rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    return graph

# Ví dụ huấn luyện và vẽ cây cho một tỷ lệ nhất định, chẳng hạn 80/20
ratio = 0.8
X_train, X_test, y_train, y_test = splits_bc[ratio]
dt_model = train_and_visualize_decision_tree(X_train, y_train, max_depth=None)
graph = visualize_tree(dt_model, feature_names=X_train.columns, class_names=['B','M'])
graph.render("decision_tree_bc_80_20", format='png', cleanup=True) # Xuất ra file ảnh


####################################
# Bước 6: Đánh giá mô hình
####################################

def evaluate_model(dt, X_test, y_test):
    y_pred = dt.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    return acc

# Đánh giá mô hình vừa huấn luyện
acc = evaluate_model(dt_model, X_test, y_test)

####################################
# Bước 7: Khảo sát ảnh hưởng của max_depth
####################################

max_depth_values = [None, 2, 3, 4, 5, 6, 7]
accuracies = []

for md in max_depth_values:
    dt_model_md = train_and_visualize_decision_tree(X_train, y_train, max_depth=md)
    acc_md = evaluate_model(dt_model_md, X_test, y_test)
    accuracies.append(acc_md)

# Vẽ biểu đồ thể hiện sự thay đổi accuracy theo max_depth
plt.figure(figsize=(6,4))
plt.plot([str(m) for m in max_depth_values], accuracies, marker='o')
plt.title("Accuracy vs. max_depth")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.show()

