import os
import numpy as np
from scipy import misc
import tensorflow as tf
import align.detect_face
import facenet
from sklearn.neighbors import KNeighborsClassifier
import cx_Oracle

gpu_memory_fraction = 0.5
threshold = 0.8     #阈值


class Person:
    def __init__(self):
        self.pid = None             # 身份证号
        self.face_coding = None     # 人脸特征向量（二进制）
        self.name = None            # 姓名
        self.gzdw = None            # 工作单位

        self.bounding_box = None        # 方框
        self.image = None               # 截取图像
        self.container_image = None     # 视频原始图像
        self.embedding = None           # 截取人脸特征向量


class Recognition:
    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.db = cx_Oracle.connect('DWD/Dwd_1234@110.90.121.17:1521/XE')
        self.identifier = Identifier(self.db)

    def add_identity(self, image, person_name):
        persons = self.detect.find_faces(image)

        if len(persons) == 1:
            person = persons[0]
            person.name = person_name
            person.embedding = self.encoder.generate_embedding(person)
            return persons

    def identify(self, image):
        persons = self.detect.find_faces(image)

        for i, person in enumerate(persons):
            # 获取截图中人脸特征向量
            person.embedding = self.encoder.generate_embedding(person)
            person.pid = self.identifier.identify(person)

            if person.pid is not None:
                cursor = self.db.cursor()
                cursor.execute("SELECT NAME, GZDW FROM ZT_RKK_RLSB WHERE pid = :id", id=person.pid)
                result = cursor.fetchone()
                if result is not None:
                    person.name, person.gzdw = result
                    print('身份证号：' + person.pid)
                    print('姓名：' + person.name)
                    print('工作单位：' + person.gzdw)

        return persons


# knn算法从数据库中查找匹配人员信息
class Identifier:
    def __init__(self, db):
        cursor = db.cursor()
        cursor.execute("SELECT pid, face_coding FROM ZT_RKK_RLSB")
        results = cursor.fetchall()

        # frombuffer将数据库中存储的二进制转换回特征向量
        train_face_codings = np.array([np.frombuffer(row[1].read()) for row in results])
        self.class_pids = [row[0] for row in results]

        # KNN：人脸特征向量作为训练集，pid作为标签
        self.knn_model = KNeighborsClassifier(n_neighbors=1)
        self.knn_model.fit(train_face_codings, self.class_pids)

    def identify(self, person):
        if person.embedding is not None:
            distances, indices = self.knn_model.kneighbors([person.embedding])
            # print(distances[0][0])
            # 若最近匹配的人脸欧氏距离小于阈值，则判断为同一个人
            if distances[0][0] <= threshold:
                # print(distances[0][0])
                # print(self.class_pids[int(indices[0][0])])
                return self.class_pids[int(indices[0][0])]
            else:
                return None


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            # load_model
            facenet.load_model('D:/Desktop/智慧岛/face_recognition/20180402-114759/20180402-114759.pb')

    def generate_embedding(self, person):
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(person.image)
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    minsize = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        persons = []
        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        for bb in bounding_boxes:
            person = Person()
            person.container_image = image
            person.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            person.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            person.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            person.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            person.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[person.bounding_box[1]:person.bounding_box[3], person.bounding_box[0]:person.bounding_box[2], :]
            person.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            persons.append(person)

        return persons
