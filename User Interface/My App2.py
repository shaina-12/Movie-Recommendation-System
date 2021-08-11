# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 19:44:53 2021

@author: hp
"""

from tkinter import *
import os
from csv import writer
import time
import pandas as pd
from PIL import Image, ImageTk
from tkinter import messagebox  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from surprise import Dataset, Reader, accuracy, SVD
from surprise.model_selection import train_test_split
import sys

class Content(object):
    def __init__(self, latent_content):
        """top N memory based: content
        """
        self.content = latent_content

    def predict_top_n(self, query, n=5):
        a_1 = np.array(self.content.loc[query]).reshape(1, -1)
        content = cosine_similarity(self.content, a_1).reshape(-1)
        dictDf = {'content': content}
        similar = pd.DataFrame(dictDf, index=self.content.index)
        similar.sort_values('content', ascending=False, inplace=True)
        return similar.head(n+1)[1:].index.tolist()


class Collaborative(object):

    def __init__(self, latent_collab):
        """top N memory based: collaborative
        """
        self.collab = latent_collab

    def predict_top_n(self, query, n=5):

        a_1 = np.array(self.collab.loc[query]).reshape(1, -1)
        collab = cosine_similarity(self.collab, a_1).reshape(-1)
        dictDf = {'collaborative': collab}
        similar = pd.DataFrame(dictDf, index=self.collab.index)
        similar.sort_values('collaborative', ascending=False, inplace=True)
        return similar.head(n+1)[1:].index.tolist()


class Hybrid(object):

    def __init__(self, latent_content, latent_collab):
        """top N memory based: hybrid
        """
        self.content = latent_content
        self.collab = latent_collab

    def predict_top_n(self, query, n=5):

        a_1 = np.array(self.content.loc[query]).reshape(1, -1)
        a_2 = np.array(self.collab.loc[query]).reshape(1, -1)
        content = cosine_similarity(self.content, a_1).reshape(-1)
        collaborative = cosine_similarity(self.collab, a_2).reshape(-1)
        hybrid = ((content + collaborative)/2.0)
        # a data frame of movies based on similarity to query
        dictDf = {'hybrid': hybrid}
        similar = pd.DataFrame(dictDf, index=self.content.index)
        similar.sort_values('hybrid', ascending=False, inplace=True)
        return similar.head(n+1)[1:].index.tolist()


class Model(object):

    def __init__(self, algorithm):
        """top N for a particular user
        """
        self.algo = algorithm

    def predict_top_n_user(self, ui, ratings_f, Mapping_file, n=5):
        if ui in ratings_f.userId.unique():
            ui_list = ratings_f[ratings_f.userId == ui].movieId.tolist()
            d = {k: v for k, v in Mapping_file.items() if not v in ui_list}
            predictedL = []
            for i, j in d.items():
                predicted = self.algo.predict(ui, j)
                predictedL.append((i, predicted[3]))
                pdf = pd.DataFrame(predictedL, columns=['movies', 'ratings'])
                pdf.sort_values('ratings', ascending=False, inplace=True)
                pdf.set_index('movies', inplace=True)
            return pdf.head().index.tolist()
        else:
            print("User Id does not exist in the list!")
            return None


def give_recommendaions1(latent_matrix_1_df, latent_matrix_2_df, ch):
    a = Content(latent_matrix_1_df)
    b = Collaborative(latent_matrix_2_df)
    c = Hybrid(latent_matrix_1_df, latent_matrix_2_df)
    query = m2.get()
    l = []
    if(ch == 1):
        l=a.predict_top_n(query)
    elif(ch == 2):
        l=b.predict_top_n(query)
    else:
        l=c.predict_top_n(query)
    df = pd.read_csv('C://Users//hp//Desktop//MyMovies.csv')
    # l1=[]
    l2 = []
    # l3=[]
    print(l)
    flag = 0
    for i in range(len(l)):
        il2 = df[((df['title'] == l[i]))].index.tolist()
        print(il2)
        if(len(il2)==0):
            flag = 1
            l2.append(-1)
        else:
            p2 = il2[0]
            # l1.append(df.loc[c].iat[0])
            print(p2)
            l2.append(df.loc[p2].iat[2])
            # l3.append(df.loc[c].iat[3])
    global c3
    c3 = Toplevel(o1)
    c3.title('Small Recommendations')
    c3.geometry('1600x225')
    c3.configure(bg='#dfdfdf')
    l4 = []
    l4.append(tuple(['Title','Genres']))
    for i in range(len(l)):
        if(l2[i] != -1):
            l4.append(tuple([l[i],l2[i]]))
    Label(c3, text='', bg='#dfdfdf').grid(row=0, column=0, sticky=E)
    Label(c3, text='Limited Movies Recommendations', font=('Helvetica bold', 18),fg='white', bg='black').grid(row=1, column=0, sticky=E)
    if(flag == 1):
        Label(c3, text='Sorry! There is not much to recommend.', fg='red', bg='#dfdfdf').grid(row=2, column=0, sticky=E)
    else:
        Label(c3, text='', bg='#dfdfdf').grid(row=2, column=0, sticky=E)
    for i in range(len(l4)):
        for j in range(len(l4[0])):
            e = Entry(c3, width=120, fg='blue', font=('Helvetica bold', 11), bg='light blue')
            e.grid(row=i+3, column=j)
            e.insert(END, l4[i][j])


def give_recommendaions2(algorithm, ratings_f, Mapping_file, ch):
    d = Model(algorithm)
    user_id = username_verify.get()
    print(user_id)
    l = d.predict_top_n_user(float(user_id), ratings_f, Mapping_file)
    print(l)
    df = pd.read_csv('C://Users//hp//Desktop//MyMovies.csv')
    # l1=[]
    l2 = []
    # l3=[]
    """for i in range(len(l)):
        il1 = df[((df['title'] == l[i]))].index.tolist()
        print(il1)
        p = il1[0]
        print(p)
        # l1.append(df.loc[c].iat[0])
        l2.append(df.loc[p].iat[2])
        # l3.append(df.loc[c].iat[3])"""
    flag = 0
    for i in range(len(l)):
        il1 = df[((df['title'] == l[i]))].index.tolist()
        print(il1)
        if(len(il1)==0):
            flag = 1
            l2.append(-1)
        else:
            p = il1[0]
            # l1.append(df.loc[c].iat[0])
            print(p)
            l2.append(df.loc[p].iat[2])
            # l3.append(df.loc[c].iat[3])
    global c2
    c2 = Toplevel(o)
    c2.title('Recommendations')
    c2.geometry('1600x225')
    c2.configure(bg='#dfdfdf')
    l4 = []
    l4.append(tuple(['Title','Genres']))
    for i in range(len(l)):
        if(l2[i] != -1):
            l4.append(tuple([l[i],l2[i]]))
    Label(c2, text='', bg='#dfdfdf').grid(row=0, column=0, sticky=E)
    Label(c2, text='Limited Movies Recommendations', font=('Helvetica bold', 18),fg='white', bg='black').grid(row=1, column=0, sticky=E)
    if(flag == 1):
        Label(c2, text='Sorry! There is not much to recommend.', fg='red', bg='#dfdfdf').grid(row=2, column=0, sticky=E)
    else:
        Label(c2, text='', bg='#dfdfdf').grid(row=2, column=0, sticky=E)
    for i in range(len(l4)):
        for j in range(len(l4[0])):
            e = Entry(c2, width=120, fg='blue', font=('Helvetica bold', 11), bg='light blue')
            e.grid(row=i+3, column=j)
            e.insert(END, l4[i][j])
    #df = pd.read_csv('C://Users//hp//Desktop//MyMovies.csv')
    #df[((df['title'] == l[i]))].index.tolist()


def mymodel(c):
    ch = c
    movies = pd.read_csv('C://Users//hp//Desktop//movies.csv')
    tags = pd.read_csv('C://Users//hp//Desktop//tags.csv')
    ratings = pd.read_csv('C://Users//hp//Desktop//ratings.csv')
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=True)
    ratings = ratings.sort_values(by=['userId', 'movieId'], ascending=[
                                  True, True], na_position='first', inplace=False, kind='mergesort', ignore_index=True, key=None)
    ratings = ratings.dropna()
    ratings = ratings.reset_index(drop=True)
    ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 35)
    movie_list_rating = ratings_f.movieId.unique().tolist()
    movies = movies[movies.movieId.isin(movie_list_rating)]
    Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))
    tags.drop(['timestamp'], 1, inplace=True)
    ratings_f.drop(['timestamp'], 1, inplace=True)
    if(ch >= 1 and ch <= 3):
        mixed = pd.merge(movies, tags, on='movieId', how='left')
        # print(mixed.head())
        mixed.fillna("", inplace=True)
        mixed = pd.DataFrame(mixed.groupby('movieId')[
                             'tag'].apply(lambda x: "%s" % ' '.join(x)))
        Final = pd.merge(movies, mixed, on='movieId', how='left')
        Final['metadata'] = Final[['tag', 'genres']].apply(
            lambda x: ' '.join(x), axis=1)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(Final['metadata'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                                index=Final.index.tolist())
        svd = TruncatedSVD(n_components=200)
        latent_matrix = svd.fit_transform(tfidf_df)
        n = 200
        latent_matrix_1_df = pd.DataFrame(
            latent_matrix[:, 0:n], index=Final.title.tolist())
        ratings_f1 = pd.merge(movies[['movieId']],
                              ratings_f, on="movieId", how="right")
        ratings_f2 = ratings_f1.pivot(
            index='movieId', columns='userId', values='rating').fillna(0)
        svd = TruncatedSVD(n_components=200)
        latent_matrix_2 = svd.fit_transform(ratings_f2)
        latent_matrix_2_df = pd.DataFrame(
            latent_matrix_2, index=Final.title.tolist())
        give_recommendaions1(latent_matrix_1_df, latent_matrix_2_df, ch)
    else:
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings_f[['userId', 'movieId', 'rating']], reader)
        trainset, testset = train_test_split(data, test_size=.25)
        algorithm = SVD()
        algorithm.fit(trainset)
        predictions = algorithm.test(testset)
        give_recommendaions2(algorithm, ratings_f, Mapping_file, ch)


def ContentModel():
    movies = pd.read_csv('C://Users//hp//Desktop//movies.csv')
    tags = pd.read_csv('C://Users//hp//Desktop//tags.csv')
    ratings = pd.read_csv('C://Users//hp//Desktop//ratings.csv')
    movies['genres'] = movies['genres'].str.replace('|', ' ', regex=True)
    ratings = ratings.sort_values(by=['userId', 'movieId'], ascending=[
                                  True, True], na_position='first', inplace=False, kind='mergesort', ignore_index=True, key=None)
    ratings = ratings.dropna()
    ratings = ratings.reset_index(drop=True)
    ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 35)
    movie_list_rating = ratings_f.movieId.unique().tolist()
    movies = movies[movies.movieId.isin(movie_list_rating)]
    Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))
    tags.drop(['timestamp'], 1, inplace=True)
    ratings_f.drop(['timestamp'], 1, inplace=True)
    mixed = pd.merge(movies, tags, on='movieId', how='left')
    # print(mixed.head())
    mixed.fillna("", inplace=True)
    mixed = pd.DataFrame(mixed.groupby('movieId')[
                         'tag'].apply(lambda x: "%s" % ' '.join(x)))
    Final = pd.merge(movies, mixed, on='movieId', how='left')
    Final['metadata'] = Final[['tag', 'genres']].apply(
        lambda x: ' '.join(x), axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(Final['metadata'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=Final.index.tolist())
    svd = TruncatedSVD(n_components=200)
    latent_matrix = svd.fit_transform(tfidf_df)
    n = 200
    latent_matrix_1_df = pd.DataFrame(
        latent_matrix[:, 0:n], index=Final.title.tolist())
    a = Content(latent_matrix_1_df)
    query = m.get()
    l = a.predict_top_n(query)
    df = pd.read_csv('C://Users//hp//Desktop//MyMovies.csv')
    # l1=[]
    l2 = []
    # l3=[]
    """for i in range(len(l)):
        il = df[((df['title'] == l[i]))].index.tolist()
        print(il)
        d = il[0]
        # l1.append(df.loc[c].iat[0])
        l2.append(df.loc[d].iat[2])
        # l3.append(df.loc[c].iat[3])"""
    flag = 0
    for i in range(len(l)):
        il = df[((df['title'] == l[i]))].index.tolist()
        print(il)
        if(len(il)==0):
            flag = 1
            l2.append(-1)
        else:
            d = il[0]
            # l1.append(df.loc[c].iat[0])
            print(d)
            l2.append(df.loc[d].iat[2])
            # l3.append(df.loc[c].iat[3])
    global c1
    c1 = Toplevel(c)
    c1.title('Small Recommendations')
    c1.geometry('1600x225')
    c1.configure(bg='#dfdfdf')
    """l4 = [('Title', 'Genres'),
          tuple([l[0], l2[0]]),
          tuple([l[1], l2[1]]),
          tuple([l[2], l2[2]]),
          tuple([l[3], l2[3]]),
          tuple([l[4], l2[4]])]"""
    l4 = []
    l4.append(tuple(['Title','Genres']))
    for i in range(len(l)):
        if(l2[i] != -1):
            l4.append(tuple([l[i],l2[i]]))
    Label(c1, text='', bg='#dfdfdf').grid(row=0, column=0, sticky=E)
    Label(c1, text='Limited Movies Recommendations', font=('Helvetica bold', 18),fg='white', bg='black').grid(row=1, column=0, sticky=E)
    #Label(c1, text='', bg='#dfdfdf').grid(row=2, column=0, sticky=E)
    if(flag == 1):
        Label(c1, text='Sorry! There is not much to recommend.', fg='red', bg='#dfdfdf').grid(row=2, column=0, sticky=E)
    else:
        Label(c1, text='', bg='#dfdfdf').grid(row=2, column=0, sticky=E)
    for i in range(len(l4)):
        for j in range(len(l4[0])):
            e = Entry(c1, width=120, fg='blue', font=('Helvetica bold', 11), bg='light blue')
            e.grid(row=i+3, column=j)
            e.insert(END, l4[i][j])
            
def models():
    mymodel(4)
    
def models1():
    mymodel(1)
    
def models2():
    mymodel(2)
    
def models3():
    mymodel(3)
            
def others():
    global o1
    o1 = Toplevel(o)
    o1.title('Small Recommendations')
    o1.geometry("600x300")
    o1.configure(bg='#dfdfdf')
    global m2
    global m12
    m2 = StringVar()
    Label(o1, text="", bg='#dfdfdf').pack()
    Label(o1, text='Continued', font=('bold', 18),fg='black', bg='white').pack()
    Label(o1, text="", bg='#dfdfdf').pack()
    Label(o1, text="Enter the name of the movie that you liked: ",
          fg='black', font=("Bold", 10), bg='#dfdfdf').pack()
    Label(o1, text="", bg='#dfdfdf').pack()
    m12 = Entry(o1, width=80, textvariable=m2)
    m12.pack()
    Label(o1, text="", bg='#dfdfdf').pack()
    Button(o1, text="Content Based Filtering", width=30, height=1, font=('bold', 10), fg='white', bg="black", command=models1).pack()
    Label(o1, text="", bg='#dfdfdf').pack()
    Button(o1, text="Collaborative Based Filtering", width=30, height=1, font=('bold', 10), fg='white', bg="black", command=models2).pack()
    Label(o1, text="", bg='#dfdfdf').pack()
    Button(o1, text="Hybrid Based Filtering", width=30, height=1, font=('bold', 10), fg='white', bg="black", command=models3).pack()
    #o1.mainloop()


def options():
    global o
    o = Toplevel(rr_win)
    o.title('Small Recommendations')
    o.geometry("450x200")
    o.configure(bg='#dfdfdf')
    Label(o, text="", bg='#dfdfdf').pack()
    Label(o, text='Choose The Type Of Recommendation', font=('bold', 18),fg='black', bg='white').pack()
    Label(o, text="", bg='#dfdfdf').pack()
    Button(o, text="Model Filtering", width=20, height=1, font=('bold', 10), fg='white', bg="black", command=models).pack()
    Label(o, text="", bg='#dfdfdf').pack()
    Button(o, text="Other Filtering", width=20, height=1, font=('bold', 10), fg='white', bg="black", command=others).pack()
    #o.mainloop()
    
def contents():
    global c
    c = Toplevel(rr_win)
    c.title('Small Recommendations')
    c.geometry("550x150")
    c.configure(bg='#dfdfdf')
    global m
    global m1
    m = StringVar()
    Label(c, text="", bg='#dfdfdf').pack()
    Label(c, text="Enter the name of the movie that you liked: ",
          fg='black', font=("Bold", 10), bg='white').pack()
    Label(c, text="", bg='#dfdfdf').pack()
    m1 = Entry(c, width=80, textvariable=m)
    m1.pack()
    Label(c, text="", bg='#dfdfdf').pack()
    Button(c, text="Show Recommendations", width=20, height=1,
           fg='white', bg="black", command=ContentModel).pack()
    #c.mainloop()

def recommend():
    us = username_verify.get()
    data1 = pd.read_csv('C://Users//hp//Desktop//ratings.csv')
    data1 = data1.sort_values(by=['userId', 'movieId'], ascending=[True, True], na_position='first', inplace=False, kind='mergesort', ignore_index=True, key=None)
    data1.drop(['timestamp'], 1, inplace=True)
    data1 = data1.dropna()
    data1 = data1.reset_index(drop=True)
    data2 = data1.groupby('userId')['movieId'].count().reset_index(name='Movies Count')
    print(data2)
    ip = data2[((data2['userId'] == float(us)))].index.tolist()
    print(ip)
    c = ip[0]
    e = data2.loc[c].iat[1]
    if(e < 35):
        contents()
    else:
        options()

def register():
    global reg_win
    reg_win = Toplevel(main_win)
    reg_win.title("My Movies - Register")
    reg_win.geometry("300x300")
    reg_win.configure(bg='#dfdfdf')
    global username
    global password
    global username_entry
    global password_entry
    username = StringVar()
    password = StringVar()
    Label(reg_win, text="", bg ='#dfdfdf').pack()
    Label(reg_win, text="Registeration", fg='black', bg="#dfdfdf", font=("Bold", 18)).pack()
    Label(reg_win, text="", bg ='#dfdfdf').pack()
    Label(reg_win, text="Enter userid and password in numeric form", fg='red', bg ='#dfdfdf').pack()
    Label(reg_win, text="", bg ='#dfdfdf').pack()
    username_lable = Label(reg_win, text="Username * ", fg='black', bg="#dfdfdf", font=("Bold", 12))
    username_lable.pack()
    username_entry = Entry(reg_win, textvariable=username)
    username_entry.pack()
    Label(reg_win, text="", bg ='#dfdfdf').pack()
    password_lable = Label(reg_win, text="Password * ", fg='black', bg="#dfdfdf",font=("Bold", 12))
    password_lable.pack()
    password_entry = Entry(reg_win, textvariable=password, show='*')
    password_entry.pack()
    Label(reg_win, text="", bg ='#dfdfdf').pack()
    Button(reg_win, text="Submit", width=10, height=1, fg='white', bg="black", font=("Bold", 10), command=register_user).pack()

def login():
    global login_win
    login_win = Toplevel(main_win)
    login_win.title("My Movies - Login")
    login_win.geometry("300x300")
    login_win.configure(bg='#dfdfdf')
    Label(login_win, text="", bg ='#dfdfdf').pack()
    Label(login_win, text="Login", fg='black', bg="#dfdfdf", font=("Bold", 18)).pack()
    Label(login_win, text="", bg ='#dfdfdf').pack()
 
    global username_verify
    global password_verify
 
    username_verify = StringVar()
    password_verify = StringVar()
 
    global username_login_entry
    global password_login_entry
 
    Label(login_win, text="Username * ", fg='black', bg ='#dfdfdf', font=("Bold", 16)).pack()
    username_login_entry = Entry(login_win, textvariable=username_verify)
    username_login_entry.pack()
    Label(login_win, text="", bg ='#dfdfdf').pack()
    Label(login_win, text="Password * ", fg='black', bg ='#dfdfdf', font=("Bold", 16)).pack()
    password_login_entry = Entry(login_win, textvariable=password_verify, show= '*')
    password_login_entry.pack()
    Label(login_win, text="", bg ='#dfdfdf').pack()
    Button(login_win, text="Submit", fg='white', bg="black", width=10, height=1, font=("Bold", 10), command = login_verify).pack()
    
    
def register_user():
    username_info = username.get()
    password_info = password.get()
    List=[username_info,password_info]
    print(username_info)
    print(password_info)
    with open('C:/Users/hp/Desktop/User.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(List)        
        f_object.close()
    file = open(username_info, "w")
    file.write(username_info + "\n")
    file.write(password_info)
    file.close()
    #username_entry.delete(0, END)
    #password_entry.delete(0, END) 
    formality()

def login_verify():
    username1 = username_verify.get()
    password1 = password_verify.get()
    #username_login_entry.delete(0, END)
    #password_login_entry.delete(0, END)
 
    list_of_files = os.listdir()
    if username1 in list_of_files:
        file1 = open(username1, "r")
        verify = file1.read().splitlines()
        if password1 in verify:
            rate_and_recommend()
 
        else:
            password_not_recognised()
 
    else:
        user_not_found()

def form_exit():
    a2=a.get()
    b2=b.get()
    c2=c.get()
    d2=d.get()
    e2=e.get()
    f2=f.get()
    g2=g.get()
    h2=h.get()
    i2=i.get()
    j2=j.get()
    uname = username.get()
    l1 = [uname,1,a2,964980868]
    l2 = [uname,57,b2,964980868]
    l3 = [uname,196,c2,964980868]
    l4 = [uname,216,d2,964980868]
    l5 = [uname,292,e2,964980868]
    l6 = [uname,1036,f2,964980868]
    l7 = [uname,1196,g2,964980868]
    l8 = [uname,1250,h2,964980868]
    l9 = [uname,1307,i2,964980868]
    l10 = [uname,1704,j2,964980868]
    with open('C:/Users/hp/Desktop/ratings.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(l1)   
        writer_object.writerow(l2) 
        writer_object.writerow(l3) 
        writer_object.writerow(l4) 
        writer_object.writerow(l5) 
        writer_object.writerow(l6) 
        writer_object.writerow(l7) 
        writer_object.writerow(l8) 
        writer_object.writerow(l9) 
        writer_object.writerow(l10) 
        f_object.close()
    form_win.destroy()
    
def formality():
    #root.withdraw()
    global form_win
    form_win = Toplevel(reg_win)
    form_win.title("Formality")
    form_win.geometry("700x600")
    form_win.configure(bg='#dfdfdf')
    global a
    global b
    global c
    global d
    global e
    global f
    global g
    global h
    global i
    global j
    global a1
    global b1
    global c1
    global d1
    global e1
    global f1
    global g1
    global h1
    global i1
    global j1
    a=DoubleVar()
    b=DoubleVar()
    c=DoubleVar()
    d=DoubleVar()
    e=DoubleVar()
    f=DoubleVar()
    g=DoubleVar()
    h=DoubleVar()
    i=DoubleVar()
    j=DoubleVar()
    Label(form_win, text="Please Rate The Following Movies: ", fg='black', font=("Bold", 18),bg ='white').pack()
    Label(form_win, text="", bg ='#cfe3f2').pack()
    Label(form_win, text="Toy Story (1995)", fg='blue', font=("Bold", 11),bg ='#dfdfdf').place(x=100,y=50)
    a1 = Scale(form_win, variable = a, from_ = 1, to = 5, bg='#fffa9b', orient = HORIZONTAL)
    a1.place(x=475,y=50)
    Label(form_win, text="Home For The Holidays (1995)", fg='blue', font=("Bold", 11),bg ='#dfdfdf').place(x=100,y=100)
    b1 = Scale(form_win, variable = b, from_ = 1, to = 5, bg='#fffa9b', orient = HORIZONTAL)
    b1.place(x=475,y=100)
    Label(form_win, text="Species (1995)", fg='blue', font=("Bold", 11),bg ='#dfdfdf').place(x=100,y=150)
    c1 = Scale(form_win, variable = c, from_ = 1, to = 5, bg='#fffa9b', orient = HORIZONTAL)
    c1.place(x=475,y=150)
    Label(form_win, text="Billy Maidson (1995)", fg='blue', font=("Bold", 11),bg ='#dfdfdf').place(x=100,y=200)
    d1 = Scale(form_win, variable = d, from_ = 1, to = 5, bg='#fffa9b', orient = HORIZONTAL)
    d1.place(x=475,y=200)
    Label(form_win, text="Outbreak (1995)", fg='blue', font=("Bold", 11),bg ='#dfdfdf').place(x=100,y=250)
    e1 = Scale(form_win, variable = e, from_ = 1, to = 5, bg='#fffa9b', orient = HORIZONTAL)
    e1.place(x=475,y=250)
    Label(form_win, text="Die Hard (1995)", fg='blue', font=("Bold", 11),bg ='#dfdfdf').place(x=100,y=300)
    f1 = Scale(form_win, variable = f, from_ = 1, to = 5, bg='#fffa9b', orient = HORIZONTAL)
    f1.place(x=475,y=300)
    Label(form_win, text="Star Wars:Episode V-The Empire Strikes Back (1980)", fg='blue', font=("Bold", 11),bg ='#dfdfdf').place(x=100,y=350)
    g1 = Scale(form_win, variable = g, from_ = 1, to = 5, bg='#fffa9b', orient = HORIZONTAL)
    g1.place(x=475,y=350)
    Label(form_win, text="Ran (1985)", fg='blue', font=("Bold", 11),bg ='#dfdfdf').place(x=100,y=400)
    h1 = Scale(form_win, variable = h, from_ = 1, to = 5, bg='#fffa9b', orient = HORIZONTAL)
    h1.place(x=475,y=400)
    Label(form_win, text="When Harry Met Sally... (1986)", fg='blue', font=("Bold", 11),bg ='#dfdfdf').place(x=100,y=450)
    i1 = Scale(form_win, variable = i, from_ = 1, to = 5, bg='#fffa9b', orient = HORIZONTAL)
    i1.place(x=475,y=450)
    Label(form_win, text="Good Will Hunting (1997)", fg='blue', font=("Bold", 11),bg ='#dfdfdf').place(x=100,y=500)
    j1 = Scale(form_win, variable = j, from_ = 1, to = 5, bg='#fffa9b', orient = HORIZONTAL)
    j1.place(x=475,y=500)
    Button(form_win, text="Done", width=10, height=1, fg='white', bg="black", command=form_exit).place(x=550,y=550)

def password_not_recognised():
    messagebox.showwarning("Try Again","Invalid Password")
    

def user_not_found():
    messagebox.showwarning("Try Again","User Not Found")
    
   
def reg_found():
    messagebox.showwarning("Already Rated","The movie that you have typed in is already rated by you.")    

def add_movies(ui,rat,til):
    a = pd.read_csv('C://Users//hp//Desktop//movies.csv')
    il = a[((a['title'] == til))].index.tolist()
    c = il[0]
    e = a.loc[c].iat[0]
    l = [ui,e,rat,964980868]
    with open('C:/Users/hp/Desktop/ratings.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(l)        
        f_object.close()
    Label(rate_win, text="", bg ='#dfdfdf').pack()
    Label(rate_win, text="Ratings Added Successfully", fg='green', bg ='#dfdfdf').pack()
    
def rate_cs():
    ui = username_verify.get()
    rat = r.get()
    til = mn.get()
    data1 = pd.read_csv('C://Users//hp//Desktop//ratings.csv')
    data2 = pd.read_csv('C://Users//hp//Desktop//movies.csv')
    data3 = pd.merge(data1, data2, how='inner', on=['movieId'])
    data3=data3.sort_values(by=['userId','movieId'],ascending=[True,True],na_position = 'first',inplace=False, kind='mergesort', ignore_index=True, key=None)
    data3.drop(['timestamp','genres'],1, inplace=True)
    find = ((data3['userId'] == float(ui)) & (data3['title'] == til)).any()
    print(find)
    if(find == True):
        reg_found()
    else:
        add_movies(ui,rat,til)
        
def list_of_movies():
    global lm
    lm = Toplevel(rate_win)
    lm.title('List of Movies')
    lm.geometry('1000x200')
    lm.configure(bg='#dfdfdf')
    sh = pd.read_csv('C://Users//hp//Desktop//movies.csv')
    li1 = sh['title'].tolist()
    li2 = sh['genres'].tolist()
    v = Scrollbar(lm)
    v.pack(side = RIGHT, fill = Y)
    t = Text(lm, width = 15, height = 15, wrap = NONE, yscrollcommand = v.set)
    t.insert(END,'Format: title'+' -----> '+'genres\n')
    n = len(li1)
    for i in range(n):
        t.insert(END,li1[i]+' -----> '+li2[i]+'\n')
    t.pack(side=TOP, fill=X)
    v.config(command=t.yview)
    
    
def rate():
    global rate_win
    rate_win = Toplevel(rr_win)
    rate_win.title("Menu 2")
    rate_win.geometry("600x400")
    rate_win.configure(bg='#dfdfdf')
    global movie_name
    global mn
    global r
    global r1
    mn = StringVar()
    r = DoubleVar()
    Label(rate_win, text="", bg ='#dfdfdf').pack()
    #Label(rate_win, text="", bg ='light green').pack()
    Label(rate_win, text="Ratings", fg='black', font=("Bold", 18),bg ='white').pack()
    Label(rate_win, text="", bg ='#dfdfdf').pack()
    Label(rate_win, text="Enter The Name Of The Movie That You Want To Rate: ", fg='black', font=("Bold", 10),bg ='#dfdfdf').pack()
    Label(rate_win, text="", bg ='#dfdfdf').pack()
    movie_name = Entry(rate_win, width=80, textvariable=mn)
    movie_name.pack()
    Label(rate_win, text="", bg ='#dfdfdf').pack()
    Label(rate_win, text="Add The Ratings: ", fg='black', font=("Bold", 10),bg ='#dfdfdf').pack()
    Label(rate_win, text="", bg ='#dfdfdf').pack()
    r1 = Scale(rate_win, variable = r, from_ = 1, to = 5, bg='#fffa9b', orient = HORIZONTAL).pack()
    Label(rate_win, text="", bg ='#dfdfdf').pack()
    Button(rate_win, text="See List", width=10, height=1, fg='white', bg="black", command=list_of_movies).pack()
    Label(rate_win, text="", bg ='#dfdfdf').pack()
    Button(rate_win, text="Done", width=10, height=1, fg='white', bg="black", command=rate_cs).pack()

"""def recommend():
    return 0;"""

def rate_and_recommend():
    global rr_win
    rr_win = Toplevel(login_win)
    rr_win.title("Menu 2")
    rr_win.geometry("450x300")
    rr_win.configure(bg='#dfdfdf')
    Label(rr_win, text="", bg ='#dfdfdf').pack()
    Label(rr_win, text="Welcome", fg='red', font=("Bold", 24),bg ='#fffa9b').pack()
    Label(rr_win, text="", bg ='#dfdfdf').pack()
    Label(rr_win, text="", bg ='#dfdfdf').pack()
    Button(rr_win, text="Rate It", width=10, height=1, fg='white', bg="black", font=("Bold", 16), command=rate).pack()
    Label(rr_win, text="", bg ='#dfdfdf').pack()
    Label(rr_win, text="", bg ='#dfdfdf').pack()
    Button(rr_win, text="Recommend", width=15, height=1, fg='white', bg="black", font=("Bold", 16), command=recommend).pack()
    Label(rr_win, text="", bg ='#dfdfdf').pack()
    #rr_win.mainloop()

def main_account_screen():
    global main_win
    main_win = Tk()
    main_win.title("My Movies")
    main_win.geometry("500x421")
    bg = PhotoImage(file = "C:/Users/hp/Desktop/C.png")
    canvas1 = Canvas( main_win, width = 500,height = 421)
    canvas1.pack(fill = "both", expand = True)
    canvas1.create_image( 0, 0, image = bg, anchor = NW)
    b1 = Button(main_win, text="Log In", font=("Bold", 16), fg='black',bg='white' , command = login)
    b2 = Button(main_win, text="Register", font=("Bold", 16), fg='black', bg='white', command = register)
    button1_canvas = canvas1.create_window( 220, 150, anchor = "nw", window = b1)
    button2_canvas = canvas1.create_window( 210, 250,anchor = "nw", window = b2)
    mainloop()

if __name__ == "__main__":
    main_account_screen()
