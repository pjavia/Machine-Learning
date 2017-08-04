from sklearn import svm
from preprocessing import process

X, Y, v_x, v_y, t_x, t_y = process()

total = len(v_y)

clf = svm.SVC()
clf.fit(X, Y)
v_y_ = clf.predict(v_x)
p = np.sum(v_y_ == np.array(v_y))
accuracy = p/float(total)
print accuracy

