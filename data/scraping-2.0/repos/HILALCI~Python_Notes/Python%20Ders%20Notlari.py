"""

                                                                        PYTHON DERS NOTLARI      



Bu kodlar python v3 icindir.Diger surumlerde farklilik gosterebilir.
!!!!!! Jython,Cython farkli dilleride destekliyor ayrinti icin https://en.wikipedia.org/wiki/Python_(programming_language)

"""
"""
https://www.python.org/ #Kurulum yapmak icin python resmi web sitesinden indirebilirsiniz.
https://docs.python.org/3/  #Bu siteden dokumanlara ulasabilirsiniz.Ayni zamada arama yaparak yine sizlere asagidaki pydoc sorgulara alternatif bir sorgu yapabilirsiniz iyi hazirlanmis bir dokumandir.
https://docs.python.org/3/tutorial/index.html #Python kendi tutorial'dir.Yine guzel hazirlanmis yine bu tutorial'a bakabilirsiniz.
"""

""" Dokuman yazilari icin kullanilir.""" #help() metodu kullanildiginda yorumlar gozukur.
#Fonksiyonlar icin ise tanimladiktan def'ten sonra baslatilmalidir.
''' Buda cok satirli yorum icin kullaniliyor.''' #Multi line comment'tir yukaridaki ile benzer sekilde belirtilen alan icerisi yorumdur.Alanin disindaki kodlarlar calistirilir.Alanin icindekiler atlanir.
#Bu tek satir yorumdu (single line comment) #Satir bittiginde yine sonraki satirdaki kod calisir.Fakat satir bitene kadar yorumdur bu yorumlar calisirken atlanir.Hata ayiklama icin kullanabilirsin.
## #Bu  sekilde de gordum fakat farklarını bilmiyorum.
python -m pydoc #cmd calistirdiginda metadocumentation calistirir.
python -m pydoc math #Kutuphane veya methodlarin  ismini direkt yazabilirsin.
python -m pydoc -k ssl # -k keyword arastirmasi yapmak icin ekleniyor bu sayede konu ile alakali method ve kutuphaneler gosteriyor.
python -m pydoc setuptools.ssl_support #Fakat ayrinti bakmak icin -k yazmadan  yaziyoruz.
python -m pydoc -p 1000 #-p port kisaltmasi istenilen portta daha sonra b enter ladigimizda browserdan lazilan portta dokumantasyon aciliyor.
python -m pydoc -b #Yukarida port girmistik burada portu kendi ayarliyor browserda dokumantasyon aciyor. !!!! Fakat her ikisindede q enter layip islemleri bitirmen lazim server acık kalir. !!!!
python -m pydoc -w json #-w write islemi icin ilgili dosyada bir html dokumantasyon olusturuyor bu ornekte json alakali bir dokumantasyon yazilacak.
pyhton dosyaadi.py  #cmd de PYTHON codelari calistirliyor.
#Kutuphane ve modulleri arastirmak icin dir() ve help() fonksiyonlarini kullanabilirsin.Fakat bazi kutuphaneleri veya methodlari sorgulamadan once importlamaniz gerekir. 
import math #math kutuphanesi kullanmak icin import etmemiz gerekiyor.
help(math) #Bu sorguyla yine math kutuphanesi hakkinda bilgi edinebiliriz.
dir(math) #math kutuphanesindeki methodlari gormek icin kullanabiliriz.
print(dir(math)) #Python shell sorgulama yapiliyor dir() fakat bu sekilde ekrana yazdirabilir.
help(math.log2) #Bu setilde math kutuphanesindeki log2 methodu hakinnda bilgi icin bu sekiyde yazilabilir.Bunu diger methodlarda da ve dir() icin de yapabiliriz.
from math import * #Alt satirdada goruldugu gibi math. gosteriminden kurtulmak icin bu sekilde import etmeliyiz.
help(log2)
#Sunu yapabilirsiniz shell yazildiginda calisiyorsa print() islemini unutmussunuz demektir. !!!  Bazi kodlarda  print yazmamis olabilirim siz eger bu durumu yasarsaniz print() ekleyin. !!!

sudo apt install python3 python3-pip build-essential python3-dev #Pythonla alakli ne varsa tek satirda indirmek icin diebiliriz.
python -V #Hangi versiyonun kurulu oldugunu gormek icin
python --version #Bu komut ile yine versioynunu gorebilirsiniz.
python3 -V yada python3 --version #Eger v2 varsa v3 sorgusu icin python3 olarak yazmamaiz gerekir.
sudo apt install idle #Teminalde calistirdigimizda Linux sistemlerede  idle kurmak icin (Best IDE ever)
idle #IDLE calistirmak icin
idle x.py #Direkt olarak dosya ile acmak icin
python #Console sadece python yazarsak python shell calisiyor ve dosyadan degil direkt olarak yine python kodlari yazip calistirabiliriz.
python3 #Python 2 de kurulu ise pyhton yazdigimizda v2 acacaktir.Bu sekilde  python v3 un shellini acariz.
exit() #Python Shell cikmak icin kullanilir.
python calistir.py #Tek bir surum varsa v3 de olsa calistirinca hata veriyor bu sekilde calistiriniz.
python3 calistir.py #Console ile calistrimak icin python yazarsak python v2 calistiriliyor. v3 icin python3 kullaniyoruz.
pip install modul_adi #Modul adi yerine kurmak istediginiz paketlerin isimini yaziyoruz.
pip3 list #Kurulu olan paketleri listeler
pip install -r req.txt #Kendi olusturmus oldugumuz txt dosyasindaki modullerin tamamini tek satirda indirmek icin.
#txt dosyasinda sadece modul isimlerinin olmasi yetiyor.
which python3 #Terminale bu komutu girdigimizde python hangi klasor altinda calistirildigi yani dosya yolunu gorebilirsin.

#Hello World
print("Hello World") #Ekrana yazdirma islemi yapmak icin adettendir ilk yazilan kod Hello Worlddur.
ad = input("Lutfen Adinizi  Giriniz: ") #Girdi islemi icin input() kullaniriz.
print(ad) #Su bir on tanim yapiyoruz bazi islemler icin simdilik bilmeniz icin.
türkçeYazı = "Bu utf-8 test etmek içindir."
print(türkçeYazı)#Python v3 ile utf-8 destekliyor.Kodlarimizda utf-8 encoding eklenmesine gerek kalmadan kullanabilir.

#Operator : Burda bazi matematiksel operatorleri ele aldik sonradan yine operatorler anlatilmaya devm edecek.
s=2 # = Matematikteki gibi denklik icin degil atama(assignment) islemi icin kullanilir.Esitlik kontrolunu == ile yapiyoruz.Esit olmadigi durumu != (not equal) bu yuzden esitlik hep sonra yazilir.
a=3 #Degiskenleri isledigimizde bircok tipte atama yapildigini gosterecegiz.
s *= 1  #Islemden sonra = getirldiginde once islem yapiliyor sonra atama yapiliyor.  
s += 1  #Matematiksel olarak yaptigimiz tum islemlerle yine ayni formda islemler yapilabilir. /= , %= ...
S ** 3  #Us alma islemi iki tane carpma ** ile yapilir.
s**0.5  #Karekok icin 0.5 
s**(1/3) #Kupkok icin parantezli 1/3 yapilir
x % y  #Kalan Bulmak icin % kullanilir.Mod operatorudur.Matematikteki gosterimi x mod y dir.
a // s  #Kalnsiz degerli tam bolme islemi icin // kullanilir.Virgulden sonrasini gostermez.Katsayıyı bulur.
21//5 #Bu islemi yaptigimizda sonucu ekrana yazdigimizda 4 yazar.Yani 21 bolunen , 5 bolen ,4 bolum , 1 kalan  yani bu islemle bolumu buluruz.



#Variable
import keyword
print(keyword.kwlist) #Reserved keywords yani onceden tanimli olan kelimeleri yazdirmak icin bu iki satiri kullaniyoruz.
#Degisken tanimlarken reserved keywords verilmiyor ayni zamanda _ haricinde operatorler yazilmiyor.Bu karisiklik olmamasi icin bu sekilde tasarlanmis.Sayiyla basliyamaz fakat ikinci karakterden sonradan sayi yazilabilir.
#Ilk karakter _ ve harf disinda bir karakter yazilamaz.Orn : 2x yanlis bir degisken tanimidir.
#Degiskenler genelde kucuk harfle baslar bu genel yazim kulturunde boyledir.
#Bir baska yazim kulturude ilk kelimeden sonraki ilk harf buyuk yazilir.Virgulden sonra bir satir bosluk yazilir.
#Daha bircok yazim kultur vardir arastirabilirsiniz.Fakat bunlari yapip yapmamak size kalmis ben bir standart olmasi icin kulture uyuyorum.
#Python3 Turkce karakterler ile degisken isimleri verilebiliyor.Fakat yinede Turkce karakter ile yazmamiz daha iyi olacaktir.Python2 desteklenmiyor.
#NOT : Degisken tanimlarken dikkatli olun cunku sonradan yanlis eklediginiz bir karakterde python ayri bir degisken olarak gorecektir.  
#Clean Code yazmak ve kodun daha sonra siz dahil baktiginda anlamli olmasi icin kisa tanim olacak sekilde degiskenler yaziniz.Ornek : a = 10 yerine sayi = 10 veya on = 10 yazmaniz daha dogrudur.Fakat intOn_TamsayiOn = 10 gibi gereksiz uzatmadan anlasilir kisalikta olmalidir.

He= 'Salih'
Yo= 'Taze'  
He+" "+Yo   #Bu islem gibi carpma da  yapilabilir.Yazilari birbirine eklemek icin.He+Yo= 'Salih Taze' olarak gozukecek." " bosluk icin eger eklemessek bitisik yazar.
s= flout(s) #Bu islem tam sayiyi ondalikli sayiya donusturuyor.Tip donusumudur(Typecasting).  
int (5.5)   #Bu islemde ondalikli sayiyi tam sayiya donusturuyor. 
str (112)   #Bu islem sayiyi string e donusturuyor.Onu bir karakter gibi algiliyor
x = "258963247"
int (x)     #Stringe bir tam sayiya ceviriyor.
x= "35.14"
str (x)     #Stringe bir ondalikli sayiya ceviriyor.Yukaridaki duzeltme ancak ondalikli ise ondalikli sayiya cevrilebilir.Tam sayiyiyida cevirir.
paragraf = """
Bu paragraf yorum degil
Paragraf gibi alt alta yazilabilir
"""
#Str farkli yazim sekillerinde  yapmak icin kullanilabilir.

j == i #Pythonda kok -1 degeri olan i degeri j ile gosterilir.
a= 3+5j
print(a.real) #Real kismini ekrana yazdirmak istersek.Ornege gore 3.0 yazcak.
print(a.imag) #Imaginer kismi yazdirmak icin kullaniyoruz.Ornege gore 5.0 yazacak.
type(a) # Sorguyu attigimizda bu <class 'complex'> sekilde cikar yani tipi komplextir.int float degil farklidir.
#Complex sayilari float a ceviremessin hata cverir.
a=100
complex(a) #Complex tihine donusturmek icin sonuc  (100+0j) bu sekilde cikar.
hex(x) # int girilen sayiyi Hexadecimal civirir.
a=None #Degeri daha belirlenmemis degisken olarak belirtilir.
a = True #Boolean deger buyuk harfle ilk basliyor.C de kucuk harfle true yanlis yazilmasin pyhton True veya False olarak yaziliyor.
x = str(5.4)
print(int(float(x))) #Direkt int cevrildiginde hata aliyoruz.Once float sonra int cevrilmeli.

if "var" in globals()
if "var" in locals() 
#Degisken isminin global veya local tanimli olup olmadigini kontrol  edebilirsin.

liste= ["Merhaba",3.14,...] #listeleme stringlerden farki degistirilebilmesi 
liste=list() #Bos bir liste olusturur.
len(liste) #Bu komut listenin kac elemani oldugunu gosterir.
list[1] #Bu komutla ikinci elemani gostermis oluruz.Sirlama 0 dan baslar.
list[-1] #Son elemani getirir.
list(::-1) #Listeyi sondan basa sirala.
liste1+liste2 #Birlesim kumesi mantigi oluyor.
liste[1]= 2 #Bu ikinci elemanin yerine yazilan karakterle degistiriyor.
liste[:2] #Bu 3. elemana degil liste elemaninda 2`ye kadar al demek.Ornek list=[1,2,3,4,5]  olsun cikan deger [1,2] olacak
liste[3:] #Bu liste elemaninda 3`e kadar al.
liste[:2]= [6,8] #Elemani degisecek Ornekteki gibi cikan liste [6,8,3,4,5] olacak.
liste.append("Salih") #Bu komut listeye eleman ekler.
liste= [11,5,6,7,8,9,4,5,6]
liste += [1,2,3,100] #append gibi elemanlari boylede ekleyebilirsin.Sonuc [11, 5, 6, 7, 8, 9, 4, 5, 6, 1, 2, 3, 100] olcak.
liste.pop() #Bu kodla parantez bossa son elemani listeden siler.
liste.pop(x) # x. elemani siler.
liste.sort() #Sirlamaya yariyor kucukten buyuge dogru sayilari siralar.Kelimeleri alafabeye gore sirlar.
liste.sort(reverse = True) #Buyukten kucuge sirlar reverse tende anlasilabilecegi gibi.
##Bu ornek te siralamanin kalici olmasina bakiyoruz.
liste = [1,6,5,4,3,2,7]
print(liste)
print(sorted(liste)) #Sorted kullanildiginda siralama kalici degildir.
print(liste) #Tekrar yazildiginda liste ilk hali ile kalir.
liste.sort() #sort ise kalici olarak listeyi sirali hali ile degistirir.
print(liste) #Yazdirdigimmizda siralanmis halini goruruz.
##
#Siralarken ayni tipte degiller ise hata verir.
list[0][0] #Ic ice listelenlerde 1. elmanin 1. elemanini gosterir.
liste.insert(2,"Salih'in listesi") #İlk eklenecek sira 2 elemandan sonra ne eklemek istiyorsak onu yapiyoruz.str de sayida girilebilir.
liste.remove(x) #Listeden x bulup siliyor.Eger birden fazlaysa ilk buldugunu siliyor digeri kaliyor dikkat!
liste.index(x) #Listede x kacinci sirada oldugunu bulur.
liste.count(x) #Listede x kac tane varsa gosterir.
liste.extend(liste2) #Listeleri birlestirir eleman olarak.append tek 1 eleman olarak ekler yani l1.extend(l2)=[1,2,3,4,5,6] --- l1.append(l2)=[1,2,3,[4,5,6]] farki bu
liste.clear() #Listedeki tum elemanlari siler.
del liste #Listeyi siler.
del liste[x] #x. elemani siler. 
liste=liste2.copy()  #Listeyi kopyalar ama deep copy(Derin copyalama) yapıyor 2 olan degisiklik 1 etkilemiyecek. l1=l2 dersek shadow copy(Sig kopyalama) degisiiklikler ikisinide etkileyecek.
liste.reverse() #Adindanda anlasilacagi gibi liste tersten baslatip siralar.
del(liste[x]) #listenin x elemanini siler

linp =  input("Lutfen sayilar giriniz: ") #Virgulle birden fazla deger girildiginde
liste = linp.split(",") #Virgul ayrilanlari bir listeye atabilirsin.
print(liste)

demet=(1,1,2,3) #Listeden farki bu degerler degismeyez yani immutable 'dir. Tuple olarak adlandirilirlar.
demet.count(1)  #Demet kac kere 1 gectigini gosterir.
demet.index(3)  #Elemanin sirasini gosteriyor.
demet =(10,11,12,14,15)
demet +=(1,2,3,4,5) #Listede oldugu gibi sonuc (10, 11, 12, 14, 15, 1, 2, 3, 4, 5) olacak.
tuple=([1,2,3],[4,5,7])
tuple[0][2]=1  #tuple degismez fakat biz listeyi degistirdik tuple[0]=[1,1,2,3,1] deseydik hata alacaktik.
t= 1,2,3 #Boyle yapsak tuple olarak kaydedecektir.
demet = "dene" , #Tek elemanli demet olusturmak icin mutlaka , eklenmeli yoksa str veya int olarak algilar.
print(sorted(demet)) #Demet ve tuplelari sortla siralayamiyoruz bu nedenle sorted kullandik. 
print(sorted(demet,reverse = True)) #Tersten siralamak icin.
del demet #Demet tamamen silmek icin

kume={1,2,3,4,5} #Bu gösterim kümeler yani set olarak adlandiriliyor.
kume={1,1,1,2,2,3,4,5} #Boyle yazilmis olsa bile yukaridaki gibi ayni elemandan bir tane olacak sekilde sirasi onemsiz olarak gosterir printle.
kume= set(liste) # set ile listeyi,tuple,stringleride kumeye cevirebiliriz.
kume.add(x) #Kumeye eleman ekliyoruz.
kume.discard(x) #Kumeden elemani siler.
kume.update("8") #Uzerinde gezinebilecegi formatlarda guncelleme yapabiliyor.
kume.update([6,7]) #Bu sekilde diger tiplerde eklemeler yapilabilir.
kume.remove(x) #Eleman silmek icin x yerine silinmek istenilen deger girilebilir.
print(kume.pop()) #Listden farkli olarak arguman verilmeden calistirilmali cunku index yok.Eleman da girilmesine musade verilmiyor.
#Silme islemini index olmadigindan rastgele yapiyor.Bu yuzden ekrana yazdirdik hangisini sildigini aramamak icin.
#Kumelerde index degerleri olmadigi icin rastgele bir yere ekleme yapar ve tekrar calistirildiginda kod degisebilir eleman yerleri dikkat.
#kume += {1,2,3,4} boyle bir islemde hata verir.
kume.clear() #Yine tum elemanlari set'ten silebiliriz.
del kume #Kume tamamen silmek icin

print(sorted(kume)) #Kume ve setleride ayni sekilde sortedla siraliyoruz.
print(sorted(kume, reverse = True)) #Tersten siralamak icin.
k1=set("bilgisayarkavramlari")
k2=set("salihtaze")
"""
Demet(tuple) : Demet elemanlari indisle gosterilebilir ve dilimlenebilir ama degistrilemez.Fonksiyonlar demet seklinde deger dondurulebilir.
Farkli degerler bir demet alrinda birlestirilebilir(packing) ve bir demet icindeki elemanlar topluca farkli degiskenlere aktarilabilir(unpacking).
Demetin listeye gore avantajli tarafi elemanlari degistirilemez oldugu icin daha korunakli bir yapiya sahip olmasi ve biraz daha hizli calismasidir.

Kumeler(set) : Kume elemanlarina idisle ulasilamaz.Dilimleme yapilmaz.Elemanlari degistirilemez.Bir elemandan en fazla 1 tnae olabilir.

Kaynak : Aksoy, A. (2018). Çocuklar için Uygulamalarla Python, Abaküs Yayınları
"""
print(k1|k2) # k1 V k2 birlesimini aliyor. or operatoru yani veya.Fakat bu operatorlerde sira onemli
k1.union(k2) # Yukaridaki ayni islem
k1.update(k2) #Birlesimi k1 atar.
print(k1-k2) # k1 - k2 yani k1 / k2 farkini aliyor.
k1.difference(k2) #Yukaridaki ile ayni islemdir.
k1.difference_update(k2) #Farkini alip k1 elemanini farktan cikanla degistiriyor.
print(k1&k2) # k1 ˄ k2 kesisimini aliyor. and operatoru yani  ve
k1.intersection(k2) #Yukaridaki ayni islemi yapar.
k1.intersection_update(k2) #Kesisimi  alip  elemanini kesimden cikanla degistiriyor.
k1.isdisjoint(k2) #Ayrik kume ise kesisimleri yoksa true degilse false doner.
k1.issubset(k2) # k1 k2 nin alt kumesi ise true degilse false
print(k1^k2) # k1 _V_ k2 yada yani exclusive or  dur.
sozluk={"bir":1} #Tanimlama ve karsliginda ne olacagini gostermesi icin.
sozluk["bir":]   #Demetteki elemani gostermek icin tanima karlilik olarak 1 gostercek. key : value 
sozluk["iki"] =2 #Tanim ve deger eklemek icin.
sozluk["bir"]=10 #Karsilik gelen degeri degistiriz ve 10 olur.
sozluk.keys() #Sadece tanimlari
sozluk.values() #Sadece karsilik gelen degerleri gosteri.
sozluk.items() #Tanim ve karsligini ikisini bir gosterir.
dict([('sape', 4139), ('guido', 4127), ('jack', 4098)]) #dict ile listeyi dictionarie cevirebiliriz. (key,value)
print(sozluk.get("bir")) #Verdigimiz key degerine karsilik gelen value get() methodu ile alabiliriz.
sozluk.pop("iki") #Yine elemani silmek icin key degerini giriyoruz.
sozluk.popitem() #Arguman vermeden cagiriyoruz. popitem() methodu ile son ekelen key ve value degerini siler.
sozluk.clear() #Tum key ve value degerlerini silebiliriz.
del sozluk #Sozlugu tamamen silmek icin

for k, v in sozluk.items(): #Anahtar ve degeri yazdirmak icin 
    print(k,v)
for i, v in enumerate(['tic', 'tac', 'toe']): #index ve deger bastirmak icin. 
    print(i, v)
x in liste #Girilen degerin listede olup olmadigini kontrol ediyor sonucunda Trua , False dondurur.Sadece listede degil demetlerde yapilabiliyor.
from collections import deque
liste2=deque([3,2,5,4,6]) #Bu sekilde listeyi  queue seklinde kaydediyor.
print(liste2)
liste2.popleft() #Bu islem pop(0) ayni fakat stackte LIFO (Last In First Out), queue FIFO (First In First Out) oldugundan kuyrugun basindan elemani cikarmamiz gerekiyor.
print(liste2)

sorted("Python Ogreniyorum") #Stringleride siralayabiliriz.
sorted("150486") #int oldugunda hata veriyor yine string olmali.

#Swap 2 variable
a = 10
b = 50
print("A= {} ,  B={}".format(a,b))
a,b = b,a  #Python ozgudur stack ile ROT_TWO() method calisiyor ve tek satirda swap islemini yapiyoruz.
print("A= {} ,  B={}".format(a,b))
x,y,z = 20,10,50 #int x= 20,y=10,z=50; //Bu sekilde C oldugu gibi tek satirda birden fazla degiskene atama yapabilirsin.Sira ile eslestiriyor. 

#Output : Cikti islemleri icin kullaniyoruz.Default olarak ekrana cikti veriyor.
print(Salih Yazilim Ogreniyor)  #Ekrana yazdirma komutudur.
print(Salih,Ingilizce,"Konusur",11)  #Ekrana birden fazla karakter bastirmak icin virgulle ayrilir.Yazilinca aralarinda bosluk olur
print(Salih\nYazilim\nProgramlar)    #Ekrana yazdirirken alt alta yazdirir.
print(Salih\tKitap\tOkur)            #Ekrana yazdirirken yan yan aralarinda 4 satir bosluklu yazadirir.
#Daha fazla : https://docs.python.org/3/reference/lexical_analysis.html?highlight=escape%20sequence
type(3.14)  #Parantez icindekinin hangi tur oldugunu gosterir.Ornekte ondalik{float} olarak gosterir.
print(01,01,1974, sep = "/")           #Ekrana yazilirken bosluk olan yere sep = "" degerin icine ne yazdiysak aralarinda o olacak.
print(Salih,Ingilizce,"Konusur",11, sep = "\n") #Kisa bir sekilde ayrilan karakterlere \n ekliyor.Yazdirirken yine alt alta yazilacak.
print(*"PYTHON")        #Degerin onune eklenen yildiz{*} degeri tek tek 1 bosluk birakarak ayirir.Fakat string olarak yazilmali {""}
print(*"TC ", sep".")
print(f"Hello  {name}") #Bu sekilde C# benzeri bir yapi ile yazdirma islemi yapabiliriz.

#f"" seklinde tum stringlere uygulayabilirsin.Bu sekilde degiskenler kullanarak Orn:replace islemni dinamik hale getirebilirsin.
pgno = 10
sayfa = f"{str(pgno)}. sayfadayiz."
print(sayfa)
#Ornegin yukaridaki ornek gibi

x = 10
y = 20
print("%i + %i = %i " %(x, y, x+y)) #C yapisina benzer sekilde de bu islemi yapabilirsin.

print(" {} {} ` nin carpimi {}`dir" .format(100,0,0)) #Format degisiiklik icin kullanilir.Parntez icindekiler onceden yazilmis({}) yerine yazilir.
print(" {1} {0} {2}" ..format(11,10,12) ) #Onden yazilmis ({}) icine yazdigimiz degerler formatin parantezindeki yazilacak sirayi gosterir.Sira 0 dan baslar.Biz  bu ornekte kucukten buyuge siralamis olduk.
"{:.1f} {:.2f} {:.3f} " .format(2.24,1.50,4.45)  # (:. f) da virgulden son kac basamak alinacagini yazdik.f in onundeki sayi ile kac basamak alincaksa yazilir.
print("{:.5}".format("Bir kismini al")) #Bu sekilde str belli bir karaktere kadar olan kismi alabiliriz.
print("{:^20}".format("Baslik")) #^ ortalamak icin ve soluna bir karakter eklemez isek bosluk birakicak verdigimiz 20 karakter boyutuna gore.
print("{:-^20}".format("Baslik")) #^ onune ekledigimiz herhangi bir karakter ile sagini ve solunu dolduracak.Fakat tek karaktere izin veriyor.
#Cikti  yandaki gibi olacak  -------Baslik-------
print("{:->20}".format("Baslik")) #Saga yasli sekilde olacak.Cikti --------------Baslik
print("{:-<20}".format("Baslik")) #Sola yasli sekilde olacak.Cikti Baslik--------------
print("{:^20d}".format(556)) #Str oldugu gibi int ayni islemleri yapilabilir.
print("{:^+20d}".format(556)) #+ pozitif sayida onunde + gozukmesi icin.C deki gibi i int olarak kabul etmiyor.
print("{:^20,d}".format(566656)) #, basamaklari , ile ayirmaya yariyor bircok karakteri kabul etmiyor.
print("{:^20.2f}".format(566.52)) #Eger nokta ile 2 basamak olacagini belirtmezsek 20 karakter old. diger virgulden sonrakileri 520000 gibi gosteriyor.

ay,gun,yil = 20,10,2023
print(ay,gun,yil, sep="/") #Sep parametresi default olarak bosluk karakteri vardir.Virgul ile ayrilan operantlari arasina istedigimiz karakteri ekliyebiliriz.

yan = "YAZI"
print(yan, end="\t") #end default olarak "\n" degerine sahiptir.Yine istedigmiz karakter ile sonlandirabiliriz.
print(yan, end="\t") #end ile Java println yerine print gibi yazdirdik bu ornekte farkli islemler yapmak mumkundur.
print("Nasil")

print(*range(11)) #Yan yana 0 dan 10 kadar tum sayilari ekrana yazdimak icin bu yontem kullanilabilir.

type(x) #Yazilan kodun ne turde oldugunu gosterir.
type(liste[i]) #Listenin elemenlarini type ile kontrol edebiliriz.

#Input : Girdi islemleri icin kullaniyoruz.Default olarak klavyeden girdi aliniyor.
input() #Kullanicinin girdilerini gormeye yariyor.
int(input("Lutfen Sayi Giriniz:")) #Girdiyi sayi olarak gormek icin 
float(input()) #Bu komutla tam sayi olmayan degerlerde ondalililarda yazilabilir.
a = input("Lutfen bir Deger Giriniz.") 
type(a) ##Default olarak string olarak okuma yapilir.Yukaridaki satirlarda tip donusumu yapmamizin nedeni budur.type() fonksiyonu kullandigimizda  str olarak goruruz.

#Logical Operator ve kontrol yapilari
#Mantik operatorleri matematikte gordugumuz mantik gibi  dogruluk  aynidir.Bu dogruluk tablosuna asina iseniz zorlanmiyacaksiniz.Tablo icin : https://en.wikipedia.org/wiki/Truth_table
a==b #A B es.Esitse True degilse False
a!=b #A B Esit degil Esitsede False farkli ise True. not equal   oldugundan once ! sonra = gelir.Yine alttakilerdede = sonra dikkat edersen.
a>b #Bu ve alttaki 3 tanede mat mantik sagliyorsa True saglamiyorsa False
a<b #Dipnot stringlerde ise alafabeye bak. 
a>=b #Bu yine hatali yazimin cok  yapildigi bir koddur hata yapmamak icin su sart kabul edebilir = hep sonradan gelir diger kullanimlarda da ornek += gibi.
a<=  
2==3 and 2==2 #False cikar cunku hepsi dogruysa dogru cikiyor.
2==3 or  2==2 #Bir tane dogru varsa True.Ikisi False ise False
not 2==2      #True ise False ,False ise True cevirir.
not(2==3 !=  2==2) #Bu sekilde xor yapabiliyoruz.
bool(0) #False doner diger tum sayilar True ayni C deki gibi.float tutulmus olsa bile 0.0 dada False doner.
print(int(True)) # 1 sonucu cikar.
print(int(False)) #0 sonucu cikar.
print(9+True) # 10 sonucu yazar.
print(9 * False) # 0 sonucu yazar.
bool("") #False doner bos degilse  True doner.bool(" ") bosluk karakteri de olsa bir karakterdir bu yuzden True doner.
l1=[1,2,3,4,5] 
l2=[1,2,3,4,5]
print(id(l1)) #Cikti 1780674395840 .Bu, bellekteki nesnenin adresidir. !!!( Pointer yani daha ayrinti icin CPython bakacagım. )!!!
print(id(l2)) #Cikti 1780723682816 
l1 == l2 #True deger doner.
l1 is l2 #False deger doner.Ustte gorumdugu fibi id yani ram adresleri  farkli oldugu icin false donuyor.
"Salih" is "Salih" #True deger doner.
10 is 10 #True deger doner.
"aa" is "a" * 2 #True deger doner.
a= "a" * 2
"aa" is a  #True deger doner.
a = "a"
"aa" is a * 2 #False deger doner.

#if-elif-else Kontrol yapilari bunlar belirli bir kosul altinda kod blogunun calismasini saglarlar.Yapisal programlamanin bir ozelligidir.

if (sayi>0):          #Kosul belirtir fakat gerkce True ise calisir False ise calismaz.
	print("Pozitif Sayi")   #Bu boslugu tab tusuyla yapabilirsin.

else:                    #Iki nokta koyarsak kosul gerkmeden If False ise calisir.Kosul eklersek oda True olunca calisir.
	print("Negatif Sayi")  	
elif (sayi==0):    #Baska bir kosul eklemek icin kullanilir.Elif ile cok fazla kosul yazilabilir.
	print("Notr Sayi")
if: elif: else:   #Siralama bu seklide olmali yoksa hata veriyor.Else ve elif kosullari tek basina calismaz.Parantez kullanmayabilirsin.

if 0: #False kabul etmek icin 0 haricinde tum sayilar True
if 2<x<20: #Bu sekilde kontrol yapilabilir.
    print("True")
a[0]=0 # a daki degisen listeyi yazdiraca esitledik.Degisiklik her ikisinide etkileyecek.
a=b[:] #Liste ilk basta ayni olacak degisiiklikler kendi degiskenleriyle sinirli kalacak etkilemiyecek
a in 4 #True yazicak listede eleman var mi kontrolunu yapiyor.

#Match-Case == Switch-Case
#Match-Case icin Python v3.10 ve ustu surumlerde desteklenmektedir.
status = x
match status:
    case 400:
        print("Bad request")
    case 404:
        print("Not found")
    case "a" | "A": #Pipe operatorunu kullanarak birden fazla kontrolu tek casede sagliyabilirsin.
        print("I'm a teapot")
    case 200:
        return "No problem"
    case _: #Default yapisinin karsiligidir.
        print("IDK")
    case ( 100,  _): #Tuple seklinde deger verebilir ve yine _ kullanabilirsin.Sadece tuple seklinde degil dic ve kwargs ... kullanabilirsin.
        print("IDK")
    case ( x, y) if x > y: #Case de kosul kontrolleri yine yapabilirsin.
        print("IDK")

#Switch-Case yapisindan farkli olarak break eklenilmesine gerek duyulmamasidir.Sadece bir tane case calisir.
#Ayrica enum ve classlarla birliktede kullanilan ornekleri vardir.


#Loop : Donguler yine belirli bir kosul dahilinde kodlarin tekrar etmesini sağlar.Yine yapisal programlamanin bir ozelligidir.
while True: #Sonsuz dongu olur.Istersek True yerine kosul ekleyebiliriz. : while blokunun basladigini belirtmek icin kullaniyoruz. : for def kullaniyoruz : altindaki tab ile bosluk biraktigimiz blogu buna gore calistiriyor.  
#Sonsun songu bazen istenmeyen bir durumdur.Bazi hatali kontrolden dolayi surekli calisir bu sonsuz calismayi durdurmak icin Ctrl+C ile durdurabiliriz.
range(1,11,2)  #1 den dahil  11'e kadar dahil degil ikiser artarak liste olusturuyoruz. range(baslangic,bitis,adim)

rg = range (0, 10, 2)
print(type(rg)) #range class'inda oldugunu donderecektir.
print(rg) #Cikti olarakta range(0, 10, 2) verecek.
print(*rg) #Yan yana direkt yazdirmak icin.

for i in range(): # i printlersek listeydeki elemanlari bastirmis olacagiz.for akis saglamasi icin liste,demet,sozluk,string gibi degerler ihtiyac duyar.range() bu yuzden var. 
#range(baslangic,bitis,adim) fakat sadece range(11) derseniz 11 bitis degeri olarak kabul eder 11 kadar olan 11 dahil degildir.Default baslangic degeri = 0 , adim degeri = 1 birer birer artar.Baslangic degeri dahildir.Adim degeri ne kadar atlama yapacagini belirtir.Adim degeri negatif olarabilir bu sefer azalan bir durum olur.
continue #Donguyu devam ettirir.
break  #Donguyu sonlandirir.
pass #Bu komut gec komutu.

ciftList = [eleman for eleman in range(11) if eleman %2 == 0] #Bu sekilde tek satirda islem yapabilirsin.
#if mantiksal operatorler ile kosullar ekleyebilirsin.Fakat else veya elif ekleme
print(ciftList)
kareList = [eleman**2 for eleman in range(11)] #Eklenecek elemana yine islemler uygulabilirsin
print(kareList)


import math # Kutuphane veya fonksiyon cagirmak icin.
dir(math)  # Ne calisabilir yazdirmak icin.
math.log10() # fonksiyonu kullunmuk icin . kullunmulı sadece import edildiyse
from time import sleep # Eger fonksiyon bu sekilde carildiysa sleep() seklinde direk calisibilir.(,)virgul ile time kutuphanesindeki diger fonksiyonlari ekleyebilirin.
x mod y #Sayının modunu bulmak için kullanilir.
x<>y #Esit değildir.
sgrt(x) #Karkok alir.
abs(x)  #Mutlak değerini alır.

from random import *
random #0 ile 1 arasında rastgele bir sayı verir.
randint( x, y ) #x ile y arasinda y dahil rastgele tam sayi verir.
uniform( x, y ) #x ile y arasinda y dahil de edebilir etmeyedebilir  ondalikli deger verir.
liste = [2,4,3,5,6]
print(choice(liste)) #choice methodu ile listeden bir elemani secebiliriz.


left(a,x) #Soldan x e kadar alır.
right(a,x) #Sağdan x e kadar alır.
value(a) #degiskeni sayiyi cevirir.
average(list) #Ortalamasini bulur.
sum(list) #Listedeki elemanlari toplamar.
date #Tarih gosterir.
time #Zamani gosterir.
exp(x) # E sayisinin x kadar kuvvetini alir. math.exp() import ettikten sonra calistirmak icin.
cos(x) #Cos x degerini alir.
pow(x,y) # x**y yani us alir.
degress(x) #Dereceyi radyana cevirir.
radians(x) #Radyani dereceye cevirir.
fabs(x)  #Fonksiyonun mutlak degerini  alir.
from time import clock #Kutuphane yukleyip clock() ilede kullanilir.
math.floor() #Ondalikli degerden onceki en buyuk tam sayi bulur.
math.ceil()  #Ondalikli degerden sonraki en buyuk tam sayi bulur.
import math as matematik #Kutuphaneyi matematik olarak tanimli calisir yapmak icin.math olarakta kullunmaya devam edebilirsin.
from math import * #Kutuphaneyi importlamis gibi olacak fakat from old icin kullanırken math.log10() yerine  direkt log10() calistirabilicem kolaylik saglayabilir.

import random
random.randrange(start, stop, step)  #PYTHON doc aldım. https://docs.python.org/3/library/random.html
random.random() #0 il 1 arasinda degerleri verir.
random.choice(a) #Parantez ici bos olamaz.Secim yapar liste olusturup secem yapilabilir.
def fonksiyon(parametre): #Kendimiz bir fonksiyon yapilabiliriz.
print("rastgele" .upper()) #Karakterleri büyük harfle yazadirir.
print("ABCD" .lower()) #Karakterleri kucuk harfle yazadirir.
print("      rastgele     "  .strip()) #Sağdan ve soldan boslugu siler.
s = "Bu yaZi YaZim KurAlLarina UYmaz."
print(s)
print(s.swapcase()) #Kucuk harfi buyultur.Buyk harfi kucultur swapcase() methodu.
s = "bu yaziin ilk harfi sadece buyuk."
print(s)
print(s.capitalize()) #Yalnizca ilk harfi buyultur diger harfler kucuk kalir ve eger onunde sayi varsa yine buyultmez.
print(s.title()) #Bu method kelimelerin ilk harflerini buyuk yazar.
print(s.center(x)) #Center tam olarak bazi degerlerde ortalamasada yazinin sagindan ve solundan bosluk birakarak ortamaya calisiyor.
s = "155005224789"
print(s.center(len(s) + 2 ,"#")) #Bu sekilde cek gibi meblanin basinda sonunda "#" ekliyor.
#center() str uzunlugundan az ise birsey yazmiyor str boyutunu gectikten sonra soldan baslayarak eklemeler yapip ortalamaya calisiyor.
print(s.ljust(len(s) + 5 ,"#")) #Centerdan farki sadece sola yaslamasi eklemeyi sadece saga yapiyor ortalamaya calismiyor.
print(s.rjust(len(s) + 5 ,"#")) #Buda yukaridaki method gibi fakat bu sefer tam tersi saga yasliyor eklemeyi sola yapiyor.
sayi = "1"
print(sayi.zfill(2)) #rjust methodu gibi soluna ekliyor fakat 0 ekliyor.Yine karakter sayisindan az veya esit ise ekleme yapmiyor.
s = "abc"
print(s.isalpha()) #Sadece alfabetik karakterler icerip icremdegini sorguluyor ve bool donderiyor.
s = "1550"
print(s.isdigit()) #Sadece rakam barindirip barindirmadigini kontrol ediyor isalpha gibi ve bool donderiyor.
s = "1550abc"
print(s.isnumeric()) #Yine sadece sayisal deger oldugunu kontrol ediyor.
print(s.isalnum()) #Bu method isalpha, isdigit birlesimi ve alfanumerik olup olmadigini kontrol ediyor.Yine bool donderiyor.
s = "abc115adad++-*/"
print(s.islower()) #Bu method harflerin kucuk oldugunu kontrol ediyor.Bool donderiyor.
s = "WWW-W3C"
print(s.isupper()) #Bu methodda ustteki methodun tam tersi buyuk harf oldugunu kontrol ediyor.Bool donderiyor.
s = "  n  "
print(s.isspace()) #Bu method tamaminin  bosluk olup olmadigini kontrol ediyor.Bu ornekte False donderecektir.
s = "a\tb\tc"
print(s.isprintable()) #Yazdirlabilir mi degil mi kontrol ediyor.Yukaridakinde False deger donduruyor.
print(s)
print("kontrol ediliyor.".istitle()) #Kelimelerin ilk harfi buyuk mu kontrol ediyor.
print(s.expandtabs(20)) #Bu method sekme bosluklarini genisletebiliyoruz 20 yerine istedigimiz bir deger girebilir.
a.find("a") #Degiskende  arama yapmar ilk buldugu a indexini  yazar.Eger bulamassa -1 donderir.
a.rfind("a") #Yukarida find method gibi fakat bu sefer sagdan basliyor aramaya
a.index("a") #find() methodu gibi index degerini donderir farki bulamassa hata mesaji verir.Belli index arasini kontrol ettirebilirsin Ornek: a.index("a",3,5)
a.rindex("a") #index() methodu gibi fark sagdan aramaya baslamasidir.
s = "A150C"
print("-".join(s)) #Araya katilip ekleme yapmak icin kullaniliyor.Ornekteki cikti A-1-5-0-C
d = ["html", "css", "js", "php"]
print("><".join(d)) #Dizinin elemanlarinin arasinada ekleme yapabiliyoruz. Ornek cikti html><css><js><php
s = "yaziyiayirparcala"
print(s.partition("r")) #Belli bir olcute gore 3 ayiriyor.Ornke cikti ('yaziyiayi', 'r', 'parcala')
print(s.partition("x")) #Eger parametre olarak verdigimiz olcut karakter bulunmuyorsa Ornek cikti ('yaziyiayirparcala', '', '')
print(s.rpartition("r")) #partition gibi parciyor ama sagdan basliyarak buda ayrimi farkli yapiyor.
print("say".count("a")) #Stringler karaktez dizileri oldugun bazi liste methodlarida calistirabiliriz.
print("ay" in "say") #Bu sekilde var olup olmadigini yine kontrol edebiliriz.True veya False deger dondurur.

#Functions or Method :  Yapisal porgramlamanin bir ozelligidir.
def fonksiyonadi(parametre,parametre2 .....): #Parametre girmedende yapabilirsin.İki noktadan anlasilabilecegi uzere alt satirlirdi bloklar olacak.
fonksiyonadi() #Fonksiyonu cegirmak icin.Parantez ici deger alabilir.
def top(a,b,c):    #Kac parametreyse o kader girdi girilmeli.
    return a+b+c   #return ile foksiyondaki islemi baska bir zamanda kullunmumizi sagliyor.Return fonksiyonu sonlandirir bu yuzden returnden sonra yazma.
def isim(ad="Salih"): #isim() deger girmeden yazarsak Salih olarak cıkacak.Eger deger verirsek o yazilcak.
isim(ad="Salih") #Fonksiyonda soyad old varsayarsak sadece ad degistirecek sirali girildigi icin bu islem ise yarayabilir.
#Default deger ile hatalarin onune gecebilirsin.
def top(*a):  #İstedigimiz kadar degisken atayabiliyor.Fakat bunu liste olarak tutuyor dikkat.for ili kullunirkan a yani degiskeni al.
def fon(*args) #Yukaridaki ayni islem yapiyor.
def fon(**kwargs) # *args demet olarak **kwargs sozluk olarak arguman aliyor.
ef top():
    global a  #Onceden tanimli degisekni degisterdimizde her yerde degisecek cunku global dedik.Nomalde fonksiyonlar local kayitli ve sadece fonksiyon calisinci degisiyor sonrasindi oncedeh tanimli ne isi o kaliyor.
liste2=[x*2 for x in liste1] #Tek satirda listedeki her elemani 2 carpabilirirz.
   
def top(a,b): yerine top= lambda a,b: a+b #Blokla tanimlamak yerini tek satirda tanimlayabiliyoruz.Calistirirken yine fonksiyon gibi top(x,y). *a calismaz.Lambda tek satirda biter blok olarak komplike degil basit calisir.
a=list(map(lambda a,b: a+b,liste1,liste2)) #map fonksiyonu ile listede gezinmemize olanak sagliyor aynen for gibi.Asilinda  
x**2 for x in range(999) #Boylede tek satirda for kullunubiliriz.

def func(f, arg):
    return f(arg)
print(func(lambda x: x+5,3)) #Bu sekilde yazilabilir.

print((lambda x: x+5)(3)) #Bu sekilde tek satirda calistirabilirsin.
toplam = lambda *x : sum(x)
#Bu yapida birden fazla deger eklenebilir.
print(toplam(5,4,3,2,1))
us = lambda x,y : x**y
#Birden fazla parametre eklenebilir.
print(us(2,6))
#lambda ifadesini return de yada parametre olarakta kullanilabiliyor farkli bir yazimda fonksiyon yazmak icin

def yaz():
    x = 6
    return x
    
print(type(yaz())) #Return ettigi degerin tipini yazar.
print(type(yaz)) #Fonksiyon tipinde oldugnu belirtir.

def yaz():
    x = 6
    print(x)

print(yaz()) #Return degeri olmagindan ekrana NONE yazdircak ek olarak.
print(type(yaz())) #Return ettigi deger olmadigindan <class 'NoneType'> yazdirir.
#Ek olarak tekrar yaz() calistirmis gibi oldugundan 6 ekrana tekrar yazacak. 

def eslenigi(scomp):
    escomp = scomp.replace('+','-')
    return scomp, escomp 

#Birden fazla degeri de dondurebiliriz.Tuple olarak return eder.
x, y = eslenigi("5+3i")
print(x, y)

#Default deger verip parametre almasada istenilen sekilde calismasi saglanabilir
def goster(name, capWord = False):
    if capWord:
        return name.upper()
    else:
        return name.lower()

print(goster("AHMET")) #ahmet olarak ekrana yazak cunku True parametresi eklenmemis.

def yapi(*args, **kwargs): #Tek * ile oldugunda tuple olarak tutuyor ve donderiyor. ** oldugunda key value tutabilecegi dic yapida tutuyor.
#Yukaridaki yapida key ve value deger gorene kadar hepsini args atiyacak sonrasinda gelen dic elemanlari kwargs atiyacak.
#Fakat sira ile girilecek kwarg yani dic elemanlar algilandiktan sonra key ve value deger isteyecektir args artik ekleyemez.
#Sadece bir kere * ve ** olan parametre olabilir yoksa hata olur.Bunlara ek baska parametre de olabilir.
print(type(yapi(20))) #Tek veya cok girmemin bir farki yok *args tuple donderiyor.
print(yapi(20,50,30,56)) #Bu sayede tuple methodlarini kullanabilecek veya for ile eleman uzerinde gezinebilicegiz methodda
print(type(yapi())) #**kwargs dict olarak dondu.
#Bu args kwargs yerine baska isimlerde verebilirsin.
print(yapi(name= "Ali",age = 20)) #**kwarg bu sekilde cagirabiliyoruz.Birden fazla olabildigi gibi tek bir tane key ve value girebiliriz.  


#Global ve Local
sayi = 5
def guncelle():
    sayi = 10
    return sayi

print(guncelle()) #Ekrana 10 degeri basilacak.
print(sayi) #Ekrana 5 degeri basilacak nedeni foksiyonun icindeki local ve global degiskeni etkilemiyor.

sayi = 5
def guncelle():
    global sayi #Bu degiskeni global yaptik
    sayi = 10
    return sayi

print(guncelle())
print(sayi) #Iki ciktida ayni guncellenen deger ekrana yazildi.


#OOP (Object-oriented programming) : Nesne Yonelimli Programlama 
#Nesne yonelimli programlamada fonksiyon degil method olarak adlandirilirlar.
class Araba():
    mark="Volvo"
    model="S90"
    color="White"
    tork= 1500
    
        
araba1=Araba()
print(araba1) // pointerda gosterdiği adresi yazdiriyor boyle 
print(araba1.mark) // . deyip istediğimiz fonksiyonları çağırabiliriz.

class Araba():
    def __init__(self,mark,model,color,tork): #self ilk başta yazılmak zorunda. mark="Volvo" tanimi onceden yapabilirsin. 2  "_" (alt cizgi) init 2  "_" (alt cizgi) olarak yapiyoruz.
        self.mark=mark
        self.model=model
        self.color=color
        self.tork=tork
    def info(self):
        print("Marka: {} \nModel: {} \nRenk: {} \nTork: {}".format(self.mark,self.model,self.color,self.tork))
    
araba1=Araba("Volvo","S90","White",1500) #Sirayla girilmeli hata vermiyor fakat fonksiyonlari sirasiyla cagirdigi icin tork degirini bjk yazabilir
araba2=Araba("Volvo","S60","Black",1100) #Eksik girildiginde hata veriyor.
araba1.info()
araba2.info()

class Driver(Araba): 
    def __init__(self,mark,model,color,tork,haveto):#Eger sadece kalitimla bire bir aktarip baska fonksiyon eklemiyeceksek pass yazabiliriz
        super().__init__(mark,model,color,tork) #super ile tekrar self.mark model ... tanimlamadamiza gerek kalmaz ayni fonksiyonlari kullanilirsin.
        self.haveto=haveto 
    def info(self):
        print("Marka: {} \nModel= {} \nRenk: {} \nTork= {} \nAraba Sayisi= {}".format(self.mark,self.model,self.color,self.tork,self.haveto))
    
driver1=Driver("Volvo","S60","Black",1100,5)
driver1.info()


class Ogrenci():
    def __init__(self, no, ad, soyad):
        self.no = no
        self.ad = ad
        self.soyad = soyad
    
    #og1 yazdirdigimizda id yerine yazdirmak istedigimiz bir yapi icin repr kullaniyoruz.
    def __repr__(self):
        return f"""******** Ogrenci bilgileri ***********
        Ogrenci Adi: {self.ad}
        Ogrenci Soyadi: {self.soyad}
        Ogrenci Numarasi: {self.no}
        """


og1 = Ogrenci(1568, "Ali", "Yilmaz")
print(og1)

def f5(x):
    return x+5
l2=[2,6,3]
print(list(map(f5,l2)))
#Ayni islem fakat lambda cok daha az satir kod yaziliyor.
fazla5=list(map(lambda y: y+5 ,l2)) # x fonksiyon disindada tanimliyken y sadece lambda da tanimli bu bazi hatalarin yasanmamasi icin kullanulabilir.
print(fazla5) 
# for ilede yapilabilir
faz5=[i+5 for i in l2]
print(faz5)

#Cesitli kutuphaneler ve modullerin bazilarini yine anlatmistik devam ediyoruz.
from math import * #* tum methodlari eklemek icin kullanilir.Tek tek yazmak yerine * eklenir.

#Terminalde calisma yapmak icin
import sys 
sys.argv[1] #komut satirindaki degeri string olarak aliyor. 0 index dosyanin ismi var o yuzden 1 aldik baska index alabiliriz.

#Dosya islemleri 
file = open('dosyaadi', 'w',encoding="utf-8") #Yazma islemi icin acmak istedigimizde dosyayi bu sekilde yapiyoruz. w da yaparken dikkat! ayni isimili dosya varsa o dosyayi siler tekrardan yazar.encoding eklemek zorunlu degil.
file = open('dosyaadi', 'a',encoding="utf-8") #a ile acarsak eger ayni dosya varsa silmiyecek eklemeler yapabilecegiz.
file.write('This is a test\n') #Yazma islemini yapiyoruz.
file.writelines(["Bu yaziyi yaz.","Satir\n", "Satir"]) #Writelines readlines gibi liste ile bu sefer dosyaya yazdirabiliyoruz.
#Eger tek bir str yazdirilacak ise list yerine str olarak kalabilir hata vermiyor.
file.close() #Islemler bittikten sonra kapatmamiz gerekiyor.
file = open('dosyaadi', 'r') #Okuma islemi icin acmak istedigimizde dosyayi bu sekilde yapiyoruz.

with open("dosya.txt","a+") as dosya:
    dosya.write("""Bu yapi ile paragraf ekleyebilirsin
    satir
satir yazacaktir.
    """) #Burda satir tab basilmis gibi yazdi yani ne karakter goruyorsa yapisinda onu yazdiriyor.
ekle = """f string 
kullim
    ornegi"""
with open("dosya.txt","a+") as dosya:
    dosya.write(f"{ekle}") #"" oldugu gibi yine paragraf olarak """""" arasina da ekleyebilrisin sana kalmis.
    #sadece str basina f getirip f"""{degisken}""" seklinde bir yapsisi var {} icine istedigin degiskeni ekleyip yazdirabilirsin.    
 
print(file.read(x)) #Print demezsek okunur fakat ekranda gostermez. x kadar byte oku.
print(i, end="") #for ile dosyayi yazdirdigimizda  \n silmek icin kullanabilirsin.
with open('dosyaadi', 'r',encoding="utf-8") as file: #Otomatik olarak dosyayi kapatacak. mode = 'rb' olursa read binary yani binary dosyalari okumak icin yazilir.
file.read() #Dosyanin tamamini okumak istersek.
file.readline() # Satir satir okuma yapiyor fakat bir kere print ile kullandiginizda ilk satiri sadece alir bir donguyle tum satirlari okunabilir.
file.readlines() #Tum satirlari bir liste seklinde tutmaya yariyor.
print(file.readlines()[2]) #Bu sayede 3. satiri sadece yazdirdik.Index degerlerine gore istedigimiz satiri sadece yazdirabiliriz.
file.seek(x) # x. index kadar ilerlemek icin.
print(file.tell()) #Dosya imlecinin hangi byte da oldugunu bilmek icin kullanabilirsin.
with file = open('dosyaadi', 'r+',encoding="utf-8") as file: # r+ ile hem okuma hem yazma yapabiliyoruz.
liste= satir.split(",") # , ile ayrilanlari bir eleman olaraka alicak.
open("dosyaadi","amac")   #Dosyayi acmamiza yariyor.Amac okuma yazma gibi islemleri belirtmek icin.
    amac= "w" #Yeni bir dosya yazmak icin ayni isimden baska dosya varsa siler dikkit. (write)
          "a" #Dosyanin son karakterden sonra eklemek icin silmeden kullaniliyor. (add)
file= open("dosyaadi","amac",encoding=("utf-8"))  #File bir degisken baska isimde verilebilir.utf Turkce karakter girilsin die ekleyebilirin.          
file.close() #Dosyayi kapatir.
file.write() #Acilan dosyaya yazi yazar.
icerik=file.read() #Dosya okumaya yariyor.
with open("dosya_adi.uzantisi") as dosya:
    icerik = dosya.read() #Bu sekilde yapilirsin.
print(icerik)

import csv
reader = csv.reader(file)
writer = csv.writer(file)

def scope_test():
    def do_local(): 
        spam = "local spam"  #do_local fonksiyonunda tanimli kalacaktir.

    def do_nonlocal():
        nonlocal spam
        spam = "nonlocal spam" #scope_test fonksiyonunda tanimli kalacaktir.

    def do_global():
        global spam
        spam = "global spam" #tum kodda tanimli kalacaktir.

    spam = "test spam"
    do_local()
    print("After local assignment:", spam)
    do_nonlocal()
    print("After nonlocal assignment:", spam)
    do_global()
    print("After global assignment:", spam)

scope_test()
print("In global scope:", spam)


#Hata Ayiklama(Python Exception Handling)
try: #Hatali kodlari denemek icin kullaniyoruz.
#Sadece hatali kodlari degil hata olmamasini veya programin calismaya devam etmesi icin de kullanilabilir hata ayiklamalar.
    a=5/0
except: #Tum hatalarda calisacak eger ozel bir hatada calismasini istersek belirtmemiz lazim asagida ornegi var.
    print("Hata var")
except ZeroDivisionError : # Birden fazla hata yazilabilir yazim (eror,eror,...) seklinde olmali
    print("Hata sayiyi  0 bolemezsiniz.")
    print("Zoraki calisan kod")

try: 
    a=5/0
except ZeroDivisionError as hata : 
    print("Hata sayiyi  0 bolemezsiniz.")
    print(hata) #Bu yapi ile kisaca hata ne oldugu yazdirilabilir.

def hatagoster(s): #Bu fonksiyonla kendimiz hata mesaji gonderebilir denemedim ama oyla umuyorum ki kendi hatalarimizi olusturabiliriz.
    if type(s) != str:
        raise ValueError("Lutfen string tipinde deger girin")
print(hatagoster(252))
try: #Calistirilacak kod bu blokda yazilir.
except: #Hata alinirsa calisir.
else: #Hata alinmassa calisir.
finally: #Hata olsada olmasada daima  calisir.

#Hata ayiklama farkli bir yolla yapmak icin. 
def faktoriyel(x):
    pass

inp = int(input("Lufen sayi giriniz: "))
if inp < 0:
    raise Exception("Sayi 0 dan kucuk olamaz.")
else:
    faktoriyel(inp)


try: 
    a=5/0
except ZeroDivisionError as hata : 
    print("Hata sayiyi  0 bolemezsiniz.")
    print(hata)
else: #Else ile adim adim ayri ayri hata ayiklama yapilabilir.
    try: 
        x/0 =5/0
    except: 
    print("hata")

while True:
    try:
        x= int(input("Lutfen tam sayi giriniz: "))
        a=5/x
        print(a)
    except ZeroDivisionError as hata : 
        print("Hata: ",hata)
        print("Lutfen tekrar deneyiniz.")
        continue #continue yapisi ile islemlerin devam etmesi saglanabilir.
    except:
        print("Hatali input girdiniz.")
        print("Lutfen tekrar deneyiniz.")
        continue
        
#While True yapilmasi yetiyor devam ediyor program bazilarinda except yapisinda pass kullanarak yapan var.
except <ERROR TO IGNORE> as e: #Bu sekile bir ignore etme yontemi tine cevap olarak verilmis.
#Fakat kullanim olarak continue kullanmak daha iyi olacaktir.Hem diger kodu okuyacak kisiler icin

#Unit Test

#cal.py
def main():
    x = int(input("Lutfen tam sayi giriniz:"))
    print("x^2 = ", karesi(x))

def karesi(n):
    return n*2


if __name__ == "__main__":
    main()

#test_cal.py
from cal import karesi

def main():
    test_karesi()

def test_karesi():
    if (karesi(2) != 4):
        print("2'nin karesi 4 degil.")

    if (karesi(3) != 9):
        print("3'nin karesi 9 degil.")


if __name__ == "__main__":
    main()
    
#test_cal.py
from cal import karesi

def main():
    test_karesi()

def test_karesi():
    assert karesi(2) == 4
    assert karesi(3) == 9


if __name__ == "__main__":
    main()

#test_cal.py    
from cal import karesi

def main():
    test_karesi()

def test_karesi():
    try:
        assert karesi(2) == 4
    except AssertionError:
        print("2'nin karesi 4 degil.")
    try:
        assert karesi(3) == 9
    except AssertionError:
        print("3'nin karesi 9 degil.")


if __name__ == "__main__":
    main()


#Pytest
pip3 install pytest

#test_cal.py
from cal import karesi

def test_karesi():
    assert karesi(2) == 4
    assert karesi(3) == 9
    assert karesi(-2) == 4
    assert karesi(-3) == 9
    assert karesi(0) == 0

pytest test_cal.py #Terminalde python olarak degil pytest ile calistiracaksin.

#test_cal.py
from cal import karesi

def test_pozitif():
    assert karesi(2) == 4
    assert karesi(3) == 9

def test_negatif():
    assert karesi(-2) == 4
    assert karesi(-3) == 9

def test_sifir():
    assert karesi(0) == 0

#Fonksiyonlara ayirmamizin nedeni tum hatali durumlari gormek icin.Digerinde ilk hata bitiriyor.

import pytest
def test_str():
    with pytest.raises(TypeError):
        karesi("something")

#Kodlarin test edilebilir olmasi icin return olmalidir.Eger methodlarinda print olarak cikti aliyor isen test edemiyeceksin.
#Bu yuzden methodlari return vererek geri deger dondurecek sekilde olustur. main kisminda print() kullan.
#Kontrol icin liste ve donguler kullanarak birden fazla deger kontrol edilebilir.
#Unutma testler kisa ve oz olmali kontrol edilebilir olmalidir.

"""
__init__.py dosyaysi olustur.

cd ../                          #Ayni klasordeyken calismadi.
pytest klasor_ismi/


#Bu sekilde init ile icerigi bos olasa paket olarak algilayacak testi tum klasorde kendisi tarayip test etcek.
"""

from functools import reduce 
# reduce map gibi liste ve demette gezinir elemanlari toplayarak devam eder. reduce(fonksiyon,liste veya demet ...)
reduce(lambda x,y : x*y ,[1,2,3,4,5]) #Bu sekilde tek satirda faktoriyel hesabi yapilabilir.
filter(lambda x : x%2==0 , liste) #filter(fonksiyon,liste veya demet ...) fakat fonksiyon boolean deger dondurmeli.
zip(liste1,liste2) # listeleri birlestirip (x,y ....) cinsinden tutmamiza yariyor. sozlukleride birlestirebiliriz.
#zip'te birlestirme islemi min eleman sayisi kadar o yuzden birinde eleman fazla ise onlari almiyacaktir dikkat !!!
#Birlestirme sira ile yapiliyor index sirasina gore set ile yapildiginda dikkat !!! sirali olmiyacaktir index olmadigi icin.
#Evet liste haricinde diger dic, tuple, set zip ile birlestirebilirsin.
enumerate(liste) #index degerlerini ve liste elemanlarini gostermek icin kullanabilirsin.
all(liste) #Tum degerler true ise true,false var ise false dondurur.
any(liste) # all tam tersi true var ise true , tum degerler false ise false donderir.
bin(x) # onluk tabandaki bir sayiyi ikilik (binary) tabanina ceviriyor.
hex(x) # 16 lik taban cevirmek icin
abs(x) # Mutlak deger alma islemi icin kullanabilirsin.
round(x) #Yuvarlama islemi yapar.
round(2.2226,3) # 2.223 cikar
max(x,y,z, ... ) #En buyuk sayiyi buluyor.Listede veya demetlerde  de calisir.
min(x,y,z, ... ) #En kucuk sayiyi buluyor.
sum(liste) #Liste veya demeteki elemanlari toplar.
pow(x,y) #Us alma islemi icin kullanilir. x taban , y us oluyor.
salih.upper() #Tum string karakterleri buyuk harfle gosterir.
salih.lower() #Tum string karakterleri kucuk harfle gosterir.
"10.1".replace(".",",") # replace ile tum karakter harfleri veya stirnigin bir kismini  degistirebiliyoruz.Bu ornekte Türkçe bir ondalik gosterim yapmis olduk.
"Start this code".startswith("Start") # Hangi karakter veya dizinle baslamasinin kontrolunu yapar sonuc boolean'dir.Buyuk kucuk harfe duyarlidir.
"Finish this code".endswith("Finish") # Hangi karakter veya dizinle bittiginin kontrolunu yapar sonuc boolean'dir.
liste= "Bunlar bizim kodlarimiz".split(" ") #Burda girdigimiz degere gore elemanlari ayirip listeye ekliyecek.
liste= "Bunlar bizim kodlarimiz".split() #Herhangi bir deger girilmez ise bosluklara gore ayirir.
"      Bosluklari siler      ".strip() #Deger vermezsek basindan ve sonundan bosluklari siler.Eger bir karakter verirsek onu siler.
#lstrip() soldakileri rstrip() sagdakileri siler.

liste=[T,B,M,M]
".".join(liste) #String elmanlarin arasina karakter eklemeye yariyor.
.count("x") # x karakteri kac defa var gosterir.
.count("x",i) # x karakteri kac defa i. indexten baslayarak var gosterir.
find("x") #Bastan baslayip ilk buldugu x degerinin indexini verir.
rfind("x") #Sondan baslayip ilk buldugu x degerinin indexini verir.Eger deger yoksa -1 der.

import time
def zaman_hesapla(fonksiyon):
    def wrapper(sayılar):
        
        
        baslama = time.time()
        sonuç =  fonksiyon(sayılar)
        bitis =  time.time()
        print(fonksiyon.__name__ + " " + str(bitis-baslama) + " saniye sürdü.")
        return sonuç
    return wrapper
 
#Bu sekilde bir kere tanimladigimiz fonksiyonu istedigimiz bir calismadan once ekleyip sonucun gosteresiliyoruz.
 
@zaman_hesapla
def kareleri_hesapla(sayılar):
    sonuç = []
    for i in sayılar:
        sonuç.append(i ** 2)
    return sonuç
@zaman_hesapla
def küpleri_hesapla(sayılar):
    sonuç = []
    for i in sayılar:
        sonuç.append(i ** 3)
    return sonuç
    
 
print(kareleri_hesapla(range(100000)))
 
print(küpleri_hesapla(range(100000)))

it=iter(liste) #iterator olusturmak icin iter() kullaniyoruz.
next(it) #listenin sonraki elemanini almak icin kullanabilirsin.

def hesap():
    for i in range(11):
        yield i**3    #bu generator ozelligi tasiyor bellekte saklamyior kullandiktan sonra isi bitiyor.
            
from datetime import datetime
print(datetime.now()) #Now dan anlasilacagi gibi suan ki zamani ve tarihi gosterir.Degiskene atanabilir. Ornek: 2020-06-12 15:08:36.811028
zaman = datetime.now()
print(zaman.year) # adindanda anlasilacagi gibi sadece yili ekrana bastirir.
print(zaman.month)
print(zaman.hour)
print(zaman.minute)
print(datetime.strftime(zaman,"%Y")) # Yukaridaki islemlerin anisisni yapar.
# "%Y" = Yil icin , "%B" = Ay icin , "%A" = Gun ismi icin Ornek: Friday , "%X" = Saat icin , "%D" = Gun bilgisi icin Ornek: 06/12/20 . Ayrintili liste icin : https://docs.python.org/3/library/time.html?#module-time
print(datetime.ctime(zaman)) #Farkli bir formatta zamani gosterir. Ornek: Fri Jun 12 15:08:36 2020
import locale
locale.setlocale(locale.LC_ALL, '') #Localden zaman cekmeye yariyor.
ileri_tarih = datetime(2050,10,10) #Bir tarih atayabiliriz
print(ileri_tarih - zaman) # Ne kadar fark oldugunuda gosterebiliriz.
saniye = datetime.timestamp(zaman) #Saniye degerini aliyoruz
tarih = datetime.fromtimestamp(saniye) #Saniyeyi tarihe ceviriyoruz
datetime.date(year,month,day) 
datetime.time(hour,minute,second) 
datetime.datetime(year,month,day,hour,minute,second) 
now = datetime.datetime.today()
print(now) # 2020-06-28 16:55:06.378933
print(now.microsecond) # 378933
moon_landing = "7/20/1969"
moon_landing_datetime = datetime.datetime.strptime(moon_landing , "%m/%d/%Y" ) # string parse time  strptime() fakat "" icine yazdigin ayni olacak yoksa bosluklar hataya sebep veriyor.

import os  #os modulunu import etmek icin
print(os.getcwd()) #os modulunu dosya konumunu ekrana yazdirir.
os.chdir(".../dosya_konumu") #os modulunun dosya konumunu degistirir.
print(os.listdir()) #os modulunun oldugu klasordeki dosyalari yazdirir.
print(os.listdir(path)) #İlgili pathdeki dosyalari listeler.
dirs = os.listdir("./") #Degiskene atarsan liste olarak dondugunden liste islemlerini yapabilirsin. ./ bulundugu klasordekilere bakar.
os.mkdir("dosya_adi") #os modulunun dosya konumuna yeni bir dosya acar.
os.makedirs("dosya_adi/yeni_dosya") #os modulunun dosya konumuna iki tane dosya aciyor ve yeni_dosya alt klasorde dosya_adi 'nin altinda kaliyor.
os.rmdir("dosya_adi/yeni_dosya") #sadece yeni dosyayi yada tek olarak yazidigimiz bir dosyayi siler.
os.removedirs("dosya_adi") #Tum dosyalari  alt dosyalari dahil siler.
os.rename("dosya_adi","dosyanin_yeni_adi") # Dosyanin ismini degistirir adindanda anlasilacagi gibi
print(os.stat("dosya_adi")) #Bazi ozelliklerini gosterir ayni saga tiklayip baktigimiz gibi
print(os.stat("dosya_adi").st_mtime) # sn cinsinde degisiklik yapildigi zamani gosterir
from datetime import datetime
print(datetime.fromtimestamp(os.stat("dosya_adi").st_mtime)) #bu sekiyde daha anlasilir bir zaman gosterir.
for i in os.walk("C:/Users/Salih/Desktop"): #dizindeki tum dosyalari konumlariyla yazdirmak icin.Ayni dosya gezgini gibi calisir "" arasina istedigin bir dosya konumunu yazabilirsin yalniz aralainda / olacak dikkat.
    print(i)
for i,j,k in os.walk("C:/Users/Salih/Desktop"): #i = Dosya Konumu , j = Klasor Ismi , k = Dosya Ismi

import sys #sys modulunu import etmemize yariyor.
sys.exit() #Cikis yapmak icin kullanilir.Calistirildiginda diger kodlar calismaz ve sonlanir.
stdin #Bu dosya input islemlerinde kullanilir.
stdout #Bu dosya cikti almak icin kullanilir.
stderr #Bu dosya hata ciktisi almak icin kullanilir.
sys.stderr.write("Bu hata mesaji \n") #Bu sekilde hata mesaji yazabilir.
sys.stderr.flush() #Hatali mesaji ekrana yazdirabiliriz.
sys.stdout.write("Mesaj\n") #Normal bir yazida yazdirabiliriz.
sys.argv[x] #komut satirindan calistirdigimazdaki giridleri almaya yariyor.

from urllib import request
resp = request.urlopen("www.co")
print(resp.code)
data =  resp.read()
html = data.decode("UTF-8")
from urllib import parse
params = {"v": "LosIGgon_KM", "t": "0m0s"}
querystring = parse.urlencode(params)
url = "https://wwww.youtube.com/watch" + "?" + querystring
resp= request.urlopen(url)
resp.isclosed()
html = resp.read().decode("utf-8")
#Kaynak:https://www.youtube.com/watch?v=LosIGgon_KM

#Consoleda bu asagidaki iki kodu yazarak indirmemiz gerekiyor.
pip3 install requests
pip3 install beautifulsoup4
import requests
from bs4 import BeautifulSoup
url= "www..co"
response = requests.get(url)   
html_icerigi  = response.content 
soup = BeautifulSoup(html_icerigi,"html.parser")
print(soup.prettify()) #Sayfa kaynagini goruntule yada view page source da gozuktugu gibi tamamini html tagleriyle ekrana yazdiracagiz.
print(soup.find_all("a")) #a etiketli olanlari ekrana yazdiriyoruz.
for i in soup.find_all("a"): #Yukaridaki islemin daha guzel ekrana yazdirmak icin kullanabilirsin.
    print(i)
    print(i.get("href")) #Sadece linkleri almak istersek bu sekiyde yapabilirsin.
    print(i.text) #Metinleri almak istersek 
print(soup.find_all("div",{"class":"tarih"})) #Bu sekilde class olarak aldiklarimizi ekrana bastiririz.
#Bilgi olsun die var guvenliligi bilmedigimden yapmadim.

#SMTP modulu guvenlik gerekcesiyle ele alinmadi fakat tekrar videsona bakabilirsin.

pip3 install Pilow #Pilow kutuphanesini kurmak icin
from PIL import Image #Pilow kutuphanesini desteklemek icin.
img = Image.open("resim.png")  #Resimi acmak icin kullaniyoruz.
img.show() #Resmi gormek icin 
img.save("resim2.png") #Resmi kaydetmek icin.
img.rotate(x).save("resim2.png") #x derece resmi dondurmek ve kaydetmek istersek.
img.convert(mode = "L") #Resmi siyah-beyaz BJK ceviriyor.
img.thumbnail(x,y) #x,y bir degiskende yazabilirsin bu islem resmin boyutunu degistirmek icin
kirp_alan = (x,y,z,t)
img.crop(kirp_alan).save("resim3.png")  #Resmin girilen kordinatlariyla kirpmaya yariyor.Kordinatlari photoscape gibi programlarla bulabilirsin.

import logging
logging.basiclog(filename = "E:\\python\\reg.log", level = logging.DEBUG, format = LOG_FORMAT , filemode= 'w')
logger = logging.getLogger()
logger.debug("This debug message.")
logger.info("This info ")
logger.warning("This warning")
logger.error("This error")
logger.critical("This critical")

#SQLite databases sqlite3 module 
import sqlite3
conn = sqlite3.connect('verim.db') #Verim adindaki database baglaniyor eger yoksa kendisi olusturyor.
cursor = conn.cursor() #Imlec atamak icin bu islemi yaptik.
def tablo_olustur():
    cursor.execute("CREATE TABLE IF NOT EXISTS kitaplik(Isim TEXT,Yazar TEXT, Sayfa_Sayisi INT)") #IF NOT EXISTS eger yoksa olusturmak icin yazdik.Bu sekilde bir tablo olusturuyoruz.
    conn.commit() #Degisiklikleri kaydetmek icin bu islemi kullaniyoruz.
    
def veri_ekle():
    cursor.execute("INSERT INTO kitaplik VALUES('And', 'Omer Seyfettin',100)")
    conn.commit()

cursor.execute("INSERT INTO kitaplik VALUES(?, ?, ?)",(isim,yazar,sayfa_sayisi)) #Bu islem kullanicidan aldigimiz degerler icin bu sekilde yapiyoruz.Foksiyordada bu degiskenleri alinacagini belirtiyoruz tabi
tablo_olustur()
veri_ekle()
def verileri_al():
    cursor.execute("SELECT * FROM kitaplik") #Kutuphanelrde oldugu gibi tum veriyi almak icin * kullandik.
    cursor.execute("SELECT Isim,Yazar FROM kitaplik") #Sadece isim ve yazar degerlerii alir.
    cursor.execute("SELECT * FROM kitaplik WHERE Yazar = 'Omer Seyfettin'") #Sadece yazari Omer Seyfettin olan aliyoruz.
    liste = conn.fetchall()
    print(liste)
    
    for row in conn.execute("SELECT * FROM kitaplik"): #Satir satir yazdirmak icin for kullanabilirsin.
        print(row)
def verileri_guncelle(sayfa_sayisi):
    cursor.execute("UPDATE kitaplik SET Sayfa_Sayisi = sayfa_sayisi WHERE Sayfa_Sayisi = 100 ") #Degisiklik yapmak istedigimizde bu islemi yapiyoruz.
    conn.commit()
def verileri_sil():
    cursor.execute("DELETE FROM kitaplik WHERE Isim = 'And' ") #Istedgimiz ozelliktekini silmek icin bu islemi yapiyoruz.
conn.close() #Baglantiyi sonlandirmak icin kullaniyoruz.

##################
#GUI Kutuphanelri#
##################
# Pyqt5 Graphic User Interface (GUI) araci ayrintili baska araclar icin: https://docs.python.org/3/faq/gui.html
#Kivy kullanmak istiyorum mobil programlama icinde yine linkten ayrintilarina ulasilabilir.

#Tkinter
#========
sudo apt install python3-tk #Kurulu degilse kurmak icin
from tkinter import * #Kutuphaneyi import etmek icin
main = Tk() #Arayuz olusturmak icin
main.mainloop() #Bu arayuzun surekli acik kalmasini sagliyor.Tk() calistigin arayuz aciliyor ve kapaniyor hizli bir sekilde bunun onune gectik.

main = Tk()
main.title("Doviz Ceviri") #Uygulama basligini degistirmek icin kullaniyoruz yoksa tk olarak kaliyor.
main.geometry("500x400") #Bu uygulama penceresinin boyutunu ayarlamamiz icin ilk deger width ikincisi height degeridir.
main.geometry("500x400+400+200") #Arti olarak eklediklerim ilk acildiginda hangi konumda olacagini belirtmek icin.x,y sirasi ile degerleri
main.resizable(True, True) #resizable(height = None, width = None) default olarak bu sayede responsive yapabiliyor fakat fullscrn ile alkali sadece
main.resizable(False, False) #Bu sayede sabit bir boyutta kalcak genisletilip kucultulemiyecek.
main.resizable(False, True) #Istenilir ise bir yonden kisitlandirilabilir.Siralama w,h icindir.
main.minsize(200, 200) #Bu satir ile kuculebilecegi en kucuk degeri veriyoruz.
main.maxsize(800, 800) #Bu satir ile buyutulebilecegi en buyuk degeri veriyoruz.
#Yukaridaki ayarlar kisit gibi gozuksede responsive bir tasarim terchinde bozulmalari onlemek icin kullanilabilir.
"""
main.grid_rowconfigure()
main.grid_columnconfigure()
#Responsive olmasi icin bu iki methodu yazilmasi hep onerilmis.Fakat parametreler farkliydi onu ayarlamalisin.
"""

main.attributes("-topmost", 1) #Bu satir ile surekli en ustte durcak baska bir window actiginizda yine ustte kalcak.alt tab yapmamk icin eklenebilir.
main.bind("<Escape>", lambda event: main.destroy()) #Bu satir ile ESC tusuna basildiginda pencereyi arayuz kapanacak.

main.attributes("-fullscreen", True) #Fullscreen olamasini saglamak icin.Fakat eger resizable minsize ve maxsize eklenmis ise duzgun calismiyor.
main.bind("<Escape>", lambda event: main.attributes("-fullscreen", False)) #Bunu ESC tusuna bastigimizda fullscreen cikmak icin yaptik.

'''
#Pack uste ve ortali bir sekilde hizalayarak bagliyor.
#Grid column ve row ile html table gibi islem yapilabiliyor.
#Place ile kordinat degerlerini verip istenilen konuma getirilebiliyor.

Label(main, text="YAZI").pack() #Bu sekilde pack ile bagliyoruz.

#Asaggidaki yapi Grid ile alt alta yapmak icindir.Yan yana  yazmak icin ise row 0 col artan yapilmalidir.
Label(main, text="YAZI").grid(row=0, column=0)
Label(main, text="YAZI").grid(row=1, column=0)
Label(main, text="YAZI").grid(row=2, column=0) 
#Grid ile merdiven yapmak icin yani degerleri degistirerek istedigimiz konuma getirebiliriz.
Label(main, text="YAZI").grid(row=0, column=0)
Label(main, text="YAZI").grid(row=1, column=1)
Label(main, text="YAZI").grid(row=2, column=2)
#Sadece simdi 3 tane ekledik sonuncuya row 9 bile desen row 2 olarak kabul edecektir.Yani 9x9 gibi istenilen yere goturmek icin farkli islemler yapilmalidir.

#Place ile  merdiven yapilisi
Label(main, text="YAZI").place(x=300, y=100)
Label(main, text="YAZI").place(x=200, y=200)
Label(main, text="YAZI").place(x=100, y=300)
'''

canvas = Canvas(main, width= 800, height= 400) #Arayuzumuzun cesitli ozelliklerini verebeliyoruz.
canvas.pack() #Canvasi baglamak icin kullaniyor.Pack disinda place ve grid var faklarini ogrenecegim.

frame_ust = Frame(main, bg= "#000000000") #Frame olusturabiliriz.
'''
frm1 = Frame(main, bg="blue", height=100, width=100, cursor="exchange") #w,h ile bouytlandirabiliriz.
#Cursor ile mouse frame alanina geldigimizde degisiklik olmasini sagliyabiliriz.
#Frame de padx , pady degeri ekleyebilirsin.
frm1.pack(side=LEFT)
#Cursorlarin listesi asagida istedigin birini kullanabilirsin.Butona yaptim frame  ekleyip guzel bir goruntu oluyor.
['plus', 'trek', 'shuttle', 'circle', 'tcross', 'fleur', 'dotbox', 'mouse', 'spider', 'arrow', 'exchange', 'target', 'man', 'watch', 'cross', 'sizing', 'pirate', 'heart', 'spraycan', 'clock', 'star']

##Not cursor ozelligini label buton ve daha bircogunda yine ekliyebiliyorsun illa frame olusturmana gerek yok.
Button(main, text="Gonder", cursor="exchange").pack(side=LEFT)
'''
frame_ust.place(relx=0.1, rely=0.1, relheight=0.1, relwidth= 0.75) #Place kullanirken rel olanlari vermemiz gerekiyor.
label1 = Label(frame_ust, text = "Luften sayi giriniz: ") #Bu sekilde label olusturabiliriz.
#lable bg vermessen beyaz olarak arka plani ayarliyacak.
label1 = Label(frame_ust, text = "Luften", font=("Times New Roman TUR", 12, "bold")) #Cesitli font ayari yapilabilir.
label1.pack() #Pack ile baglarsak ortalayip baslangica baglar.
label1.pack(padx= 10, pady= 10, side=LEFT) #Bu sekilde padding verilebilir ve side ile konum verilebilir.
'''
#Bitmap listesi asagida bunlari kullanip ayrica bir dosya icon eklemek tutmana gerek kalmiyacak. 
['hourglass', 'gray12', 'info', 'error', 'question', 'warning', 'gray25', 'gray50', 'gray75', 'questhead']
#Butona ekliebilirsin fakat yazi gostermiyor sadece icon gozukuyor.
'''
label_option = StringVar(frame_ust)
label_option.set('\t') #Bu acilir menunun boyutlandirmasi icin kullanilabilir.
#\n \t tekrar ekledigimizde saga ve alta dogru genisliyor.
#Assagidaki gibi degerleri verebiliriz.
label_open_menu = OptionMenu(
    frame_ust,
    label_option,
    "5",
    "10",
    "20",
    "50"
)
label_open_menu.pack(padx= 10, pady= 10, side=LEFT) #Side hep sola verirsek saginda ekler sola yaslanir.

mevsimList = ["kis", "ilkbahar", "yaz", "sonbahar"]
sVal= StringVar()
OptionMenu(main, sVal, *mevsimList).pack(side=LEFT) #Bu sekilde onceden liste olusturup kullanabiliriz.
sVal= StringVar(value=mevsimList[2]) #Bu islem ile default olarak secili gelecek elemani ayarliyabiliyoruz.

sVal = StringVar()
sVal.set(value=mevsimList[2]) #Yukaridaki default eleman islemini set ile yapabiliyoruz.

opt1 = OptionMenu(main, sVal, *mevsimList)
opt1.config(relief=GROOVE, padx=5, pady=5) #Degisiklikleri option .config ile yapiyoruz.OptionMenu yazdigimizda hata veriyor.
opt1.pack(side=LEFT)
print(sVal.get()) #Degeri alirken sVal opt1 den almiyoruz. 

Label(frame_ust, text = "Luften birim giriniz: ", bg= "#6ed321", font= "TimesNewRoman 12 bold").pack(padx= 10, pady= 10, side=LEFT)
#Yukaridaki yapi olmasi gerekendir cunku label degisiklik yapmadigimiz icin degiskende tutmamiza gerek yoktur.Hemde hafiza icin
Label(main, text="YAZI",wraplength=1).pack() #wraplength ile belli bir pixel sonrasi asagi atiyor 1 yerine 10 olsa alta atiyor.Pixel old.
Label(frame_ust, text = "Luften birim giriniz: ", bg= "#6ed321", font= "TimesNewRoman 12 bold").pack(padx= 10, pady= 10, side=LEFT)
var = IntVar()
radio1 = Radiobutton(frame_ust, text= "Euro", variable= var, value= 1, bg= "#6ed321", font= "TimesNewRoman 12")
radio1.pack(padx= 10, pady= 10, side=LEFT)
#Alt alta olsun istersek anchor yonler verebiliriz north west gibi
radio1 = Radiobutton(frame_ust, text= "Euro", variable= var, value= 1, bg= "#6ed321", font= "TimesNewRoman 12")
radio1.pack(padx= 10, pady= 10 , anchor=NW)
radio2 = Radiobutton(frame_ust, text= "GBP", variable= var, value= 2, bg= "#6ed321", font= "TimesNewRoman 12")
radio2.pack(padx= 10, pady= 10,  anchor=NW)
radio2 = Radiobutton(frame_ust, text= "TL", variable= var, value= 3, bg= "#6ed321", font= "TimesNewRoman 12")
radio2.pack(padx= 5, pady= 5,  anchor=NW)

var = IntVar()
rdb1 = Radiobutton(main, text="NO", variable=var, value=0, command=goster)
rdb2 = Radiobutton(main, text="YES", variable=var, value=1,  command=goster)
rdb2.pack(side=LEFT)   
rdb1.pack(side=LEFT)
# rdb2.select() #Bu islem ile secilmesini sagliyabiliyoruz.
rdb1.deselect() #Bu islem ile secimi kaldirabiliyoruz.Bu ornekte bu satir eklenmesse defult olarak seciyor secimi kaldirinca hic biri secilmemis oluyor.

var1 = IntVar() #Radio button farkli olarak her bir checkbutton icin ayri var tanimliyoruz.
checkb = Checkbutton(frame_ust, text= "Alis", variable= var1, onvalue=1, offvalue=0, bg= "#6ed321", font= "TimesNewRoman 12")
checkb.pack(padx= 5, pady= 5, anchor=NW)
var2 = IntVar()
checkb2 = Checkbutton(frame_ust, text= "Satis", variable= var2, onvalue=1, offvalue=0, bg= "#6ed321", font= "TimesNewRoman 12")
checkb2.pack(padx= 5, pady= 5,  anchor=NW)
#onvalue ve offvalue secilip secilmedigindeki degerlerini veriyoruz.
checkb.select() #Secebilirsin.
checkb2.deselect() #Secimini kaldirabilirsin.
checkb.toggle() #Bu islem not operatoru gibi dusunebilirsin var olan secimin tam tersini yapiyor.Secili ise kalidiyor degil ise seciyor.

Spinbox(main, from_=5, to=10).pack(side=LEFT) #Baslangic ve bitis degerleri verebiliyoruz from to ile
Spinbox(main, from_=10, to=100, increment=10).pack(side=LEFT) #increment ile artis miktarini verebiliyoruz.
Spinbox(main, from_=10, to=100, increment=10, justify=CENTER).pack(side=LEFT) #justify ile ortalamasini sagliyabiliriz.
#justify default olarak LEFT degerini aldigi icin sadece sag orta gibi ozel olarak belirttigimiz durumda ekliyoruz.
#state durumu icin NORMAL default degeridir.
Spinbox(main, from_=10, to=100, increment=10, justify=CENTER, state=DISABLED,disabledbackground= "blue", disabledforeground= "yellow").pack(side=LEFT)
#Buton gibi fakat disable oldugunda renk fakliligini veriyorsun disabledbackground, disabledforeground ile
Spinbox(main,from_=10, to=100, increment=0.5, format="%02.01f").pack(side=LEFT) #format ile ondalikli ifade verebiliyorsunuz.
#02 virgulden once 01 virgulden sonrasi icin.%02.03f yaptiginda virgulden sonra 3 basamak oluyor hassasiyeti arttirmak icin kullanabilirsin.
#increment degeri vermediginizde yine 1 er 1 er artiyor ondalikli kisim hep sifir oluyor.

cities = ["Sakarya","Ankara","İzmir"]
libox = listbox(self, selectmode = "multiple")
for i in cities:
    libox.insert(END,i)
libox.pack()
# expand = True  oldugunda sagina eklemeler yapiyor fill=BOTH ile uste alta eklemeler yapmaya yariyor pack icin

inp = Text(frame_ust, height= 1 , width= 10) #Ben input icin dedim ama text areadir.
inp.tag_configure("style",font=("Times New Roman TUR", 12))
inp.pack(padx= 10, pady= 10, side=LEFT)
inf = "Lutfen sayi turunde giriniz."
inp.insert(END,inf, "style") #Bu sekilde icerisine yazi ekleyip bilgilendirebiliriz.

#Input yapisi icin Entry kullanilir.Fakat text area metin paragraf islemleri icin yine kullanabilirsin.
Entry(main, show= "*").pack() #Password girmek icin kullanilabilir show ne gormek istiyorsan o yazilabilir.Fakat tek karakter gir.
#Show tek karakterden fazla girince hata vermiyor fakat ilk karakteri sadece gosteriyor o yuzden tek karakter gir.
Entry(main, justify="right").pack() #justify ile sagdan baslamasi saglanabilir.Arapca yazilmak istencek ise kullanilabilir.
Entry(main, selectforeground="yellow").pack() #Sectiginde yazin rengini degistiriyoruz selectforeground ile.
Entry(main, selectbackground="yellow").pack() #Sectiginde arka plan rengini degistiriyoruz selectbackground ile.
Entry(main, state=DISABLED).pack() #Girdgi yapilmasini istenmiyor ise eklenebilir.

sVal = StringVar(main, value="Daha sonra tekrar deneyiniz.")
Entry(main, state=DISABLED, textvariable=sVal).pack() #Textvariable ile yazi gozukmesini sagliyabiliriz.

ent1 =  Entry(main)
ent1.pack()
ent1.get() #Girilen almak icin fakat buton ile kullaniliyor genelde ve print(ent1.get()) kullanirsan terminale yazdirir.
ent1.delete(0, "end") #Bastan baslayip istedigin karakter kadar silebilirsin.
ent1.insert(0, "ekle") #Istenilen indexten baslayip ekleme yapmak icin kullanilir.
ent1.insert("end", "ekle") #Sonda ekleme yapmak icin
ent1.select_to("end") #Bu islem girilen index kadar seciyor."end" yerine 5 yazdigimizda 5. indexe kadar kismi secer.
ent1.select_range("end") #Tamamini secmek icin
ent1.select_range(2,"end") #Ikinci karakterden sonuna kadar secmek icin
print(ent1.select_present()) #select_present() secili olup olmadigini kontrol ediyor.True veya False deger donderiyor.
ent1.icursor(0) #Bu islem ise imlecin istenilen index gitmesini sagliyor.Ornekte baslangica gitmis olduk home basilmis gibi dusunebiliriz.
#Yukaridaki entry methodlari buton ile kullanilmistir direkt yazdiginizda calismamasinin nedeni bu olabilir.

btn = Button(frame_ust, text="Gonder", command=kabul) #Fakat method ustte tanimli olmali
btn.pack(anchor=S) 
btn2 = Button(frame_ust, text="Kapat", command=main.destroy) #main.destroy ile penceryi kapatabiliyoruz.
btn2.pack(anchor=S)

'''
#relief ozelligi ile buton sekil yapisini degistirebiliyor bazisinda cukur bazisinda disa cikik oluyor.Asagidakileri deneyebilirsin.
Button(main, text = "FLAT", relief = FLAT ).pack()
Button(main, text = "RAISED", relief = RAISED ).pack()
Button(main, text = "SUNKEN", relief = SUNKEN ).pack()
Button(main, text = "GROOVE", relief = GROOVE ).pack()
Button(main, text = "RIDGE", relief = RIDGE ).pack() 
#Buton veya enrty gibi  digerlerinede uygulaniyor fakat ben gorunus olarak butonlarda sadece olmasinin daha iyi olacagini dusunuyorum.
'''

Button(main, text="Gonder",activebackground="blue").pack(side=LEFT) #Mouse hover efekti gibi dusunebilirsin backte arka plan rengi degisiyor.
Button(main, text="Gonder", activeforeground="yellow").pack(side=LEFT) #fore da ise yazinin rengi degisiyor.
Button(main, text = "GROOVE", bd= 8,  relief = GROOVE ).pack() #bd border veriyor relief ile kullaninca daha iyi oluyor.Farki goruyorsun.
Button(main, text = "GROOVE",padx=10, pady=10, bd= 8,  relief = GROOVE ).pack() #padx,pady bosluk birakmasini sagliyoruz.
Button(main, text = "GROOVE", bd= 8, state=DISABLED,  relief = GROOVE ).pack() #Tiklanmamasi saglamak icin state disabled verilebilir.

def kabul():
    inf = "Hesaplama yapiliyor"
    inp.insert(END,inf, "style")
    
    if var.get(): #Secilip secilmedigini kontrol ediyoruz.
    	if var.get() == 1: #Value degeri 1 ise kontrol ettik.
            if var.get() == 1:
            bilgi = inp.get("1.0", "end").replace("\n", " ")
            #Text tamamini almak icin "1.0" ve "end" eklenmelidir.
            cevir = int(bilgi) * 10
            bilgi += " Euro " + str(cevir) + " TL dir."  
            messagebox.showinfo("Islem", bilgi )
        
#Show eror ek olarak messagebox.showerror() ve messagebox.showwarning() vardir.
def goster() :
    messagebox.showinfo("SONUC", ent1.get()) #Ilk deger arayuz title oluyor.Ikinci gostermek istedigimiz ne mesaj var ise
    messagebox.showwarning("Uyari", "Seni uyariyorum.") #Uyari mesaji icin kullanabilirsin.Yine ilki title ikincisi mesaj
    messagebox.showerror("HATA", ent1.get() + " deger yanlis tipte") #Hata mesajlari icin
    
    rsp = messagebox.askquestion("SORU","Baska birsey eklemek ister misin?") #Bu sekilde soru sorabiliyoruz.Yes or No button kendisi ekliyor.
    # print(rsp) #Terminalde cevabi gorebiliriz.
    if rsp == "yes" : #Yes butonuna tikladigin ne yapmak istiyorsak onu ekliyebiliriz.
        print("Ekliyor.")
    elif rsp == "no" :
        print("Bitti.")
        
    rsp = messagebox.askokcancel("OK or Cancel","Kapatmak ister misiniz?")
    if rsp == True :
        main.destroy()
    elif rsp == False :
        print("Hadi devam edelim.")
        
	
    rsp = messagebox.askretrycancel("Retry or Cancel","Hatali deger tekrar deneyiniz.")
    if rsp == True :
        print("Tekrar deniyorum.")
    elif rsp == False :
        main.destroy()
        
	rsp = messagebox.askyesno("Yes or No","Tamam mi devam mi?")
	if rsp == True :
    	print("Tekrar deniyorum.")
    elif rsp == False :
    	main.destroy() 
    
	rsp = messagebox.askyesnocancel("Yes or No or Cancel","Tamam mi devam mi?")
	if rsp == True :
   		print("Tekrar deniyorum.")
	elif rsp == False :
		main.destroy()
	elif rsp == "None" : #Cancel tiklandiginda None degeri donuyor.
		print("Duzenliyorum.")
    
    #Bu yapilari ayri olarak kullanmamizin bir nedeni de kendisi icon ile ne oldugunu gosteriyor olmasi tekrar icon eklemek ile ugrasmiyoruz.

Label(main, text="SAYI GIRINIZ ").pack(side=LEFT)
ent1 =  Entry(main)
ent1.pack(side=LEFT)
Button(main, text="Gonder", cursor="exchange",activebackground="blue",padx=10, pady=10, bd= 8,  relief = GROOVE, command=goster).pack(side=LEFT)

img = PhotoImage(file="gnu.png")
imgr = img.subsample(2,2) #Bu islem resimi kucultmek icin rakam buyudukce daha da kuculuyor fakat kalitede bozuluyor.
#imgr eklemek icin image atanan img yerine imgr yazmalsin yoksa yine ayni kalir kucultme yapilmamis gibi olur.
Label(main, text="yazi arkada", image=img).pack() #Bu sekilde img ekliyebilirsin.Fakat yazi arkada kalcak gozukmiyecektir.
Label(main, text="yazi arkada", image=img, compound=LEFT).pack() #compound ile yazinin soluna yada sagina resimi getirebiliyoruz.
#Resimin ustte olmasi icin TOP,altta olmasi icin BOTTOM yaziyoruz.

"""
compound = LEFT -> image will be at left side of the button
compound = RIGHT -> image will be at right side of button
compound = TOP -> image will be at top of button
compound = BOTTOM -> image will be at bottom of button

Source : https://www.geeksforgeeks.org/python-add-image-on-a-tkinter-button/
"""

menuBar = Menu(main)
fileMenu = Menu(menuBar)
fileMenu = Menu(menuBar,tearoff=0) #tearoff cizikli kisimi kaldiriyor.
fileMenu = Menu(menuBar, activebackground="yellow") #Yine bircok ozellik ekliyebiliyoruz digerleri gibi
fileMenu.add_command(label="File Open", command=file_open)
fileMenu.add_command(label="File Save", command=file_save)
fileMenu.add_checkbutton(label="Otomatik Kaydet",command=file_save) #Checkbutton ile secildikten sonra tikli kaliyor bazi islem icin kullanilabilir.
#Tabi secimi tekrar kaldirmak icin tikladiginda kalkiyor radio button gibi degil.
fileMenu.add_radiobutton(label="Public")
fileMenu.add_radiobutton(label="Private") #Bu sekilde yine radio button kullanabilirsin.Ayni stunda olduklari icin gruplamana gerek kalmiyor.
#Eger biri secili ise digerini sectiginde oburunun tiki kalkiyor.Fakat ilk basta secili degil sectikten sonra illaki biri secili kaliyor.
fileMenu.add_separator() #Bu satir ayrac gorevi goruyor oncesi ile sonrasini arasina ciziyor.
fileMenu.add_command(label="QUIT",command=close)
menuBar.add_cascade(label="File", menu=fileMenu) #label olarak atadigimiz menu adi oluyor.
main.config(menu=menuBar) #pack gibi menu arayuze bagliyoruz gibi dusunebilirsin.

#Yeni window islemleri
def newin() :
    topl = Toplevel()    
#Top level main ile bagantili pencere acmaya yariyor.Main acik kaldiginda top level pencere kapandiginda main kaliyor.
#Fakat main kapattigimizda top level pencereside kapaniyor.
main =  Tk()
Button(main, text="Yeni Pencere Ac", command=newin).pack()
main.mainloop()

def newin() :
    nwin = Tk()    
#Tk methodunu kullanarak bagimsiz ayri pencereler acilabiliyor.Biri kapandiginda digerini etkilemiyor.
main =  Tk()
Button(main, text="Yeni Pencere Ac", command=newin).pack()
main.mainloop()

def newin() :
    nwin = Tk()    
    main.destroy()
#Main destroy gibi pencereleri kapatarak tek bir pencere acip digerini kapatmis olur diger turlu tum pencereler acik kalacaktir.
#Fakat method olduklari icin birden fazla islemlerde nested method olarak yazilmalai yoksa destroy islemi yapilamiyor.
main =  Tk()
Button(main, text="Yeni Pencere Ac", command=newin).pack()
main.mainloop()


main.mainloop() #Surekli pencerinin acik kalmasini sagliyoruz yoksa kapaniyor.

#Kivy
#========
#Kurulum tamamlandiktan sonra tum islemleri yazmiyoruz.Kaynak: https://kivy.org/doc/stable/gettingstarted/installation.html

source kivy_venv/bin/activate #Virtual Environment calistirmak icin
deactivate #Virtual Environment kapatmak icin
python3 kivy_venv/share/kivy-examples/demo/showcase/main.py #Demo bakabilirsin hem araclari gosteriyor.

#Sadece ekranda yazi gozukecek basit bir arayuz tasarimi icin
from kivy.app import App
from kivy.uix.label import Label


class calisma(App):
	def build(self):
		return Label(text = "Hello World")
		
if __name__ == "__main__":
	calisma().run()
	
#Input alabildigimiz bir arayuz
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout

class giris(GridLayout):
	def __init__(self, **kwargs):
		super(giris, self).__init__(**kwargs)
		
		self.cols = 2 #cols = 1 olursa altina inputu aliyor.
		self.deger = Label(text = "Lufen °C degerini giriniz: ")
		self.add_widget(self.deger)
		self.derece = TextInput(multiline = False) #multiline false demezsek alt satira iner
		#self.pass = TextInput(multiline = False, password = True) #** seklinde gizliyecek.
		self.add_widget(self.derece)
		

class uygulama(App):
	def build(self):
		return giris()
		
if __name__ == "__main__":
	uygulama().run()	

"""
#.kv uzantili dosyamizda gorsel yapiyi ayri olusturabiliriz.

#Dosyayi ana dosyada cagirmak icin
from kivy.lang import Builder
Builder.load_file(dosyaAdi.kv)
#Eger bu islemleri yapmak istemiyorsaniz dosya adi ile ayni bir classta yaparsaniz gerek kalmaz.

#.kv uzantili dosyada yazim sekli
<label>:
	text: "Hello World"

<giris>:
	cols: 2
	
"""

#Buton arayuze eklendi
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.core.window import Window

class giris(GridLayout):
	def __init__(self, **kwargs):
		super(giris, self).__init__(**kwargs)
		
		self.cols = 2
		self.deger = Label(text = "Lufen °C degerini giriniz: ")
		self.add_widget(self.deger)
		self.derece = TextInput(multiline = False)
		self.add_widget(self.derece)
		self.cl = Button(text = "Temizle")
		self.add_widget(self.cl)
		self.send = Button(text = "Gonder")
		self.add_widget(self.send)
		

class uygulama(App):
	def build(self):
		return giris()
		
if __name__ == "__main__":
	Window.clearcolor = (0, 128, 0, 1) #RGBA renk kodu ile arka plan rengini degistirebiliriz.
	uygulama().run()
	
"""
Buton size,size_hint ve pos,pos_hint ayralanabiliyor.pos kordinati Mat oldugu gibi img oldugu gibi sol ustten degil.
_hint olanlari kullanman responsive bir yapida oluyor.
Kordinat sol alt koseden basliyor 0,0 ve 1,1 biter. 1,1 sag ust kosedir.
root.center_x, root.center_y ile merkezine alabiliriz fakat boyutlari cikarirsak tam ortali olur.
-self.width / 2, -self.height / 2 ile centerlari cikarisak tam ortali olur.
root.x, root.top -self.height  ile img baslangicina getirebilirsiniz.
"""

#Tiklama ve fare hareketleri eventleri
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.widget import Widget


class tiklama(Widget):
	def on_touch_down(self, touch): #Tiklandiginda
		print(touch)
	def on_touch_move(self, touch): #Tiklanip suruklendiginde eger basili degilse yazdirmaz
		print(touch)
	def on_touch_up(self, touch): #Tiklama sonlandiginda
		print(touch)
		#Printler terminalde kordinatlarini gosteriyor.
			
class uygulama(App):
	def build(self):
		return tiklama()
		
if __name__ == "__main__":
	Window.clearcolor = (0, 128, 0, 1)
	uygulama().run()

#BoxLayout duzeni
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.core.window import Window

class giris(BoxLayout):
	def __init__(self, **kwargs):
		super(giris, self).__init__(**kwargs)
		self.orientation = "vertical" #Dizlimin dikeyde olmasi icin
		#self.orientation = "horizontal" #Dizilimin yatayda olmasi icin
		self.padding = 10 #Kenar boslugu
		self.spacing = 10 #Butonlar arasi bosluk
		self.deger = Label(text = "Lufen °C degerini giriniz: ")
		self.add_widget(self.deger)
		self.derece = TextInput(multiline = False)
		self.add_widget(self.derece)
		self.cl = Button(text = "Temizle")
		self.add_widget(self.cl)
		self.send = Button(text = "Gonder")
		self.add_widget(self.send)
		

class uygulama(App):
	def build(self):
		return giris()
		
if __name__ == "__main__":
	Window.clearcolor = (0, 128, 0, 1) #RGBA renk kodu ile arka plan rengini degistirebiliriz.
	uygulama().run()
	

#PyQt5
#========

import sys
from PyQt5 import *
def window():
    app = QtWidgets.QApplication(sys.argv) #Uygulama olusturup sys.argv komut satirindan okuma yapmasi icin ekledik.
    pencere = QtWidgets.QWidget() #Widget olusturduk.
    pencere.setWindowTitle("Ilk Deneme") #Widget baslik ekledik.
    etiket = QtWidgets.QLabel(pencere) #Label ekleme icin 
    etiket2 = QtWidgets.QLabel(pencere) 
    etiket2.setPixmap(QtGui.QPixmap(resim.png)) #Resim eklemek icin kullaniyoruz.
    etiket.setText("Bu alanda yazi") #Yazi eklemek icin kullaniyoruz.
    button = QtWidgets.QPushButton(pencere) #Buton olusturmak icin kullanilir.
    button.setText("Tikla") 
    etiket.move(100,100) #Move ile  harekete ettiriyorsun  fakat widget boyutu degisirse bozulabilir.Responsive bir tasarim degil. 
    pencere.setGeometry(200,200,500,500) #a,b konumu icin girilen degerler.Hangi kordinattan baslayaçagi. c,d ise buyuklugunun en,boy ddegerleri.
    #                    a   b   c   d
    
    
    horizantal_box = QtWidgets.QHBoxLayout() #Yatalda bir layaout olusturmak icin QHBoxLayout kullaniyoruz.
    horizantal_box.addStretch() #Satirin basinda boslukk olmasi icin 
    horizantal_box.addWidget(kabul)
    horizantal_box.addStretch() #Kabul butonundan sonra bosluk birakiyor yanii yapilan islemden sonra bosluk birakiyor.Bu sekilde konumlari sabit kalirken sadce genisliyorlar.
    horizantal_box.addWidget(red)
    pencere.setLayout(horizantal_box) #Yaptimiz pencere widget olmasi icin bu satiri ekliyoruz.
    horizantal_box.addStretch() #Satir sonunda bosluk olmasi icin.
    vertical_box = QtWidgets.QVBoxLayout() #Dikeyde butonlari olusmasi icin QVBoxLayout kullaniyoruz.
    pencere.setLayout(vertical_box) #Ayni sekilde pencere widget olmasi icin kullaniyoruz.    
    #Farki QHBoxLayout olusturlanlar yatayda saginda vela solunda siralanirken.QVBoxLayout dikeyde yani alt alta siralanir.
    
    
    pencere.show() #Adindan anlasilacagi gibi gostermek icin ekledik.
    sys.exit(app.exec_()) #Windowsta X isaretine basana kadar calismasi icin bu satiri yazdik.
window()

def window(): 
    app = QtWidgets.QApplication(sys.argv)
    pencere = QtWidgets.QWidget()
    pencere.setWindowTitle("PyQt5 Uygulama")
    pencere.setGeometry(200,200,500,500)
    ha_box = QtWidgets.QHBoxLayout()
    next = QtWidgets.QPushButton("Ileri")
    previous = QtWidgets.QPushButton("Geri")
    ha_box.addWidget(previous)
    ha_box.addWidget(next) #Ekleme sirasi onemli ona gore siraliyor.
    av_box = QtWidgets.QVBoxLayout()
    av_box.addStretch()
    av_box.addLayout(ha_box) #Bu sekilde yatay ve dikeyi birlestiriyoruz.
    pencere.setLayout(av_box)
    pencere.show()
    sys.exit(app.exec_())
window() #Bu kodlarla altta aralarainda bosluk olacak selide ilerleme button olusturduk.
 

 class Pencere(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    def  init_ui(self):
        self.yazı_alanı = QtWidgets.QLineEdit()  #Input alani olusturmak icin QLineEdit kullaniyoruz.
        self.temizle = QtWidgets.QPushButton("Temizle")
        self.yazdir = QtWidgets.QPushButton("Goster")
        self.yazi = QtWidgets.QLabel()
        v_box = QtWidgets.QVBoxLayout()
        v_box.addWidget(self.yazı_alanı)
        v_box.addWidget(self.temizle)
        v_box.addWidget(self.yazdir)
        v_box.addWidget(self.yazi)
        v_box.addStretch()
        h_box = QtWidgets.QHBoxLayout()
        h_box.addStretch()
        h_box.addLayout(v_box)
        h_box.addStretch()
        self.setLayout(h_box)
        self.temizle.clicked.connect(self.temizleme) #clicked tiklandi bilgisini aliyoruz. connect ile methoda bagliyoruz.
        self.yazdir.clicked.connect(self.eyaz)
        self.show()
    def temizleme(self):
        self.yazı_alanı.clear() #clear ilede temizleme islemini yapiyoruz.
    def eyaz(self):
        print(self.yazı_alanı.text())
app = QtWidgets.QApplication(sys.argv)
pencere = Pencere()
sys.exit(app.exec_())
#Tek fonksiyonlada yapabiliriz onun ornegi
self.temizle.clicked.connect(self.tiklama)
self.yazdir.clicked.connect(self.tiklama)
def tiklama(self):
    sender = self.sender()  #Yapilan aksiyonu aliyoruz diebiliriz.
    if sender.text() == "Temizle": #Senderdan gelen yaziyi alip kontrol yapiyoruz.
        self.yazı_alanı.clear()
    else:
        print(self.yazı_alanı.text())

self.parola.setEchoMode(QtWidgets.QLineEdit.Password) #Girilen alandaki yazilar gizlenecek ozel karakter olacak bu sekilde.
self.yaziAlani.setText(self.inputYazi.text()) #Girdiyi alip yazdirirken setText dierek yapiyoruz.
from PyQt5.QtWidgets import QWidget,QApplication,QCheckBox,QLabel,QPushButton,QVBoxLayout
self.checkbox = QCheckBox("Anlasmayi kabul ediyorum.") #checkbox olusturmak icin yapiyoruz.
self.buton.clicked.connect(lambda : self.click(self.checkbox.isChecked())) #isChecked ile checkbox isaretlenip isaretlenmedigini kontor ediyoruz.lambda olmazsa hata verebilir genelde fonksiyon aldigi icin.
def click(self,checkbox,yazi_alani):
    if checkbox:
        yazi_alani.setText("Anlastik")
    else:
        yazi_alani.setText("Sen anlasmayi bozdun kardes")
from PyQt5.QtWidgets import QWidget,QApplication,QRadioButton,QLabel,QPushButton,QVBoxLayout
self.radio_yazisi = QLabel("Hangi dili daha çok seviyorsun ?")
self.java = QRadioButton("Java") #QRadioButton radio button olusturuyoruz.Bunlar tek secim icin kullanilir.checkbox birden fazla secim yapilabilir.
self.python = QRadioButton("Python")
self.php = QRadioButton("Php")
self.buton.clicked.connect(lambda : self.click(self.python.isChecked(),self.java.isChecked(),self.php.isChecked(),self.yazi_alani))
def click(self,python,java,php,yazi_alani):
    if python:
        azi_alani.setText("Python")
    elif php:
        yazi_alani.setText("Php")
    elif java:
        yazi_alani.setText("Java")
from PyQt5.QtWidgets import QWidget,QApplication,QTextEdit,QLabel,QPushButton,QVBoxLayout
self.yorumAlani = QTextEdit() #Bu sekilde HTML ile textarea yaptigimizin aynisini burada yapiyoruz.

import sys
import os
from PyQt5.QtWidgets import QWidget,QApplication,QTextEdit,QLabel,QPushButton,QVBoxLayout,QFileDialog,QHBoxLayout
from PyQt5.QtWidgets import QAction,qApp,QMainWindow
class Notepad(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    def init_ui(self):
        self.yazi_alani = QTextEdit()
        self.temizle = QPushButton("Temizle")
        self.ac = QPushButton("Aç")
        self.kaydet = QPushButton("Kaydet")
        h_box = QHBoxLayout()
        h_box.addWidget(self.temizle)
        h_box.addWidget(self.ac)
        h_box.addWidget(self.kaydet)
        v_box = QVBoxLayout()
        v_box.addWidget(self.yazi_alani)
        v_box.addLayout(h_box)
        self.setLayout(v_box)
        self.setWindowTitle("NotePad")
        self.temizle.clicked.connect(self.yaziyi_temizle)
        self.ac.clicked.connect(self.dosya_ac)
        self.kaydet.clicked.connect(self.dosya_kaydet)
    def yaziyi_temizle(self):
        self.yazi_alani.clear()
    def dosya_ac(self):
        dosya_ismi = QFileDialog.getOpenFileName(self,"Dosya Aç",os.getenv("HOME"))
        with open(dosya_ismi[0],"r") as file:
            self.yazi_alani.setText(file.read())
    def dosya_kaydet(self):
        dosya_ismi = QFileDialog.getSaveFileName(self,"Dosya Kaydet",os.getenv("HOME"))
        with open(dosya_ismi[0],"w") as file:
            file.write(self.yazi_alani.toPlainText())
class Menu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.pencere = Notepad()
        self.setCentralWidget(self.pencere)
        self.menuleri_olustur()
    def menuleri_olustur(self):
        menubar = self.menuBar()
        dosya = menubar.addMenu("Dosya")
        dosya_ac = QAction("Dosya Aç",self)
        dosya_ac.setShortcut("Ctrl+O")  #Kisayol tusu eklemek icin kullaniliyor.
        dosya_kaydet = QAction("Dosya Kaydet",self)
        dosya_kaydet.setShortcut("Ctrl+S")
        temizle = QAction("Dosyayı Temizle",self)
        temizle.setShortcut("Ctrl+D")
        cikis = QAction("Çıkış",self)
        cikis.setShortcut("Ctrl+Q")
        dosya.addAction(dosya_ac)
        dosya.addAction(dosya_kaydet)
        dosya.addAction(temizle)
        dosya.addAction(cikis)
        dosya.triggered.connect(self.response)
        self.setWindowTitle("Metin Editörü")
        self.show()
    def response(self,action):
        if action.text() == "Dosya Aç":
            self.pencere.dosya_ac()
        elif action.text() == "Dosya Kaydet":
            self.pencere.dosya_kaydet()
        elif action.text() == "Dosyayı Temizle":
            self.pencere.yaziyi_temizle()
        elif action.text() == "Çıkış":
            qApp.quit()   #Uygulamayi kapatmak icin.
app = QApplication(sys.argv)
menu = Menu()
sys.exit(app.exec_())


#Numpy ve veri yapilari
import numpy as np #Bu sekilde kisaltmalida kullanilabilir.Fakat ben bu sekilde ele almiyacagim.
npArray = numpy.array(liste) #Listeyi array seklinede tutuyoruz fakat numpy ile 
npArray = numpy.array([1,2,3,4,5]) #Listede oldugu gibi listeyi onceden tanimlamadan da yapabiliriz.
npArray[2,2] #Listelerde oldugu gibi [] ile elemanlara erisiyoruz fakat ic ice olanlari [][] bu gosterimin yerine ornekteki gibi yapiyoruz.[:::] bu islem yine ayni kullanabliriz.
npArray = numpy.arange(1,     9,      2) #range fonksiyonunu yine kullanabiliriz.
#                     baslangic, bitis, adim  
npArray = numpy.zeros(x) #x tane sifir olan array olusturmak icin kullanilir.
npArray = numpy.ones(x)  #x tane bir olan array olusturmak icin kullanilir.   
#numpy arraylerdeki int degerlerini float cevirir.Bu yuzden 1. veya 0. die ekrana yazilirlar.
npArray = numpy.ones((x,y)) #Bu sekilde iki boyutulu yapabiliriz.x ve y degerlerini esit olmasi zorunlu degil.
npArray = numpy.linspace(0,   10,   3) #linspace es parcaya bolme islemi yapiyor.Bir pastayi boler gibi degil fakat sirayi boler gibi dusunebiliriz.
#                     baslangic, bitis, kaca bolunecegi
npArray = numpy.eye(x) #Diyagon yapmak icin kullaniliyor eye metodu ile kaca kaclik oldunu x yerie yaziyor kare matris olusturuyor.
'''
ornek 4 4 luk bir diyagon
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
'''
npArray = numpy.random.randint(0,  10,   5) #Random degerleri array olarak saklamak istersekte bu sekilde kullanabiliriz. Ornek ciktisi [3 2 9 0 5]
#                         baslangic, bitis, kac sayi olusturulacagi
npArray = numpy.random.rand(x) #rand fonksiyonunuda kullanabiliriz 0-1 kadar x tane random sayi olusturabiliriz.
npArray = numpy.random.randn(x) # negatif olsun istersek bu sekilde yapabiliriz.Fakat sinirlar farkli tam olarak bilmiyorum.
npArray = npArray.reshape(x,x) #matris olusturmak istersek reshape yapabiliriz.Fakat arraydeki elemanlarin girilen x degeri kadar alani kaplayacak sekildeolmali bosta eleman kalinca kabul etmiyor.
npArray.max() #array deki maximum degeri almak istersek max fonksiyonununu kullaniyoruz.
npArray.argmax() #max degerin kacince indexte oldugunu bulmak icin argmax kullanabiliriz.
npArray.min() #array deki minimum degeri almak istersek min fonksiyonununu kullaniyoruz.
npArray.argmin() #min degerin kacince indexte oldugunu bulmak icin argmin kullanabiliriz.
npArray.sum() #Tum arrayleri toplamak istersek sum fonksiyonunu kullaniyoruz.
npArray.mean() #Tum arraylerin ortlamasini bulmak icinde mean fonksiyonunu  kullanabiliriz.
numpy.linalg.det(npArray) #Determinant islemi icin linalg.det() fonksiyonunu kullaniyoruz.square matrix  olmak zorunda yoksa hata veriyor.
cArray = npArray.copy() #Bu sekilde deep copy yapiyoruz yine cArray = npArray shadow copy yapabiliriz.
npArray[:x,:y] #x kadar satir y kadar stundan eleman almak icin kullanilabilir.
#satir baslangici : satir bitis , stun baslangici : stun bitis
npArray > 50 #arraydeki elemanlari kontrol edip true veya false olarak tekrar dondurur.
gecerNotlar[ gecerNotlar > 50] #True deger donenler ayni zamanda tekrar array olarak tutulacak.
npArray = npArray[npArray % 2 == 0 ] #Farkli kosullarda verilebilir.
npArray1 + npArray2 #Bu islem index sirasiyla toplar.Ornek npArray1 : [1,2,3] npArray2: [4,5,6] sonuc = [5,7,9] .Ornektede ayni eleman sayisina sahip olmalilar.
# + * / % kullanilabilir.
npArray + 5 #Her elmanin 5 fazlasi gibi islemlerde yine yapilabilir artik nasil islem yapmak istersek bircok operatoru kullanabiliriz.
numpy.sqrt(npArray) #Karekok alabiliriz.Deneyerek daha fazla islemleride yapabilriz.math kutuphanesindeki fonksiyonlarida kullanabiliriz.

#Pandas ve veri yapilari
import pandas as pd #Pandas kutuphanesini importlamak icin bu sekilde yapabiliriz.
labelList = ["Ali","Veli","Deli"]
dataList = [10,50,100]
pd.Series(data = dataList, index = labelList) #Pandas serisi olusturmak icin yazilir.
pd.Series(data = dataList) #index degerini kendisi olusturur.
pd.Series(dataList,indexList) #Bu daha kisa gösterimdir ustteki  islemin.Listeler esit sayida olmali  az veya fazla her  ise hata veriyor.
d = dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
pd.Series(d) #Sozlukleride yine pandas serilerilene donusturebiliriz.
labelList = ["Ali","Veli","Deli"]
dataList = [10,50,100]
ps1 = pd.Series(data = dataList, index = labelList)
ps2 = ps1
ps3 = ps1 + ps2 #Yine matematiksel islemler yapilabilir.
labelList2 = ["Ali","Keli","Deli"] 
dataList2 = [10,50,100]
ps4 = pd.Series(data = dataList2, index = labelList2)
ps5 = ps3 + ps4  #ps1...ps3 Keli, ps4 Veli olmayan bir veri olduguu icin  NaN veya nan (not an number kisaltmasi olarak) gösterecektir.
npArray = np.array([10,20,30,40,50])
pd.Series(npArray) #Numpy arraylerini ayni sekilde pandas serisi olarak tutabliriz.
df = pd.DataFrame(data = [(11,5,6),(7,8,9),(4,5,6)], index = ["1.","2.","3."], columns = ["Col1", "Col2", "Col3"]) #DataFrame columns degeri eklenmesi gerekiyor.Bu sekilde bir tablo yapilabilir.
df = pd.DataFrame([(11,5,6),(7,8,9),(4,5,6)],["1.","2.","3."],["Col1", "Col2", "Col3"]) #Yine pandas serilerinde  oldugu gibi siralamayi dogru yapildigi surece direkt bu sekilde de yazilabilir.
df["Col1"] #Col1 altinda siralanan verileri alabilir.Her bir kolonu yazarken yine index degerlerimizde yine yazilacak bu sekilde tablo olarak yine veriyi alabiliriz.
df["Col1"]["1."] #11 degerini gosterecek.
df.loc["1."] #Buda satiri yazdirmak icin yapabiliriz.Bu seferde columns degerlerinide yazdirarak tablo olarak görmemize yardimci oluyor.
df.loc[["1.","Total"],["Col1","Col2","Col3","Col4"]] #Burda yine istenilen tablo kisimlarini almak istersek bu sekilde yapabiliriz.
df.iloc[0] #Yukardaki islemin array index yapisiyla cagirmak istersek bu sekilde yapabiliriz.
df.iloc[0][3] #Bu sekilede yine array index yapisiyla satir ve sutunun ilgili verisini alabiliriz. 1. girdi satir icin 2. girdi stun icin 
df["Col3"] = (df["Col1"] + df["Col2"]) / 2 #Bu sekilde bir aritmetik ortalamasini alabilir.float tipinde tuttugu icin bolme islemi dogru sonuc yazacaktir.
df.loc["3."] = df.loc["1."] + df.loc["2."] #Bu sekilde excel programinda genelde yapilan tabloda toplam degerlerini bu sekilde yine yapabilir.
df[["Col1","Col2"]] #Bu sekilde iki kolonu alabiliriz. 
df["Col4"] = pd.Series(data = [10,5,25], index = ["1.","2.","3."]) #4. kolonu yine bir seri olarak tanitmaliyiz ve index degerleri ayni olmalidir.
df["Col5"] = [10,5,25] #index sayisi kadarlik bir listeyide yeni bir stun olarak ekleyebilirsin.
df["Avg"] = ((df["Col1"] + df["Col2"] + df["Col3"]) / 3) #Bu sekilde yapilabilir.
df.loc["Total"] = df.loc["1."] + df.loc["2."] + df.loc["3."] #Kolon ekledigimiz gibi satirda ekliyebiliriz.
df.drop("Col3",1, inplace = True) #Kolon tablodan silmek icin bu islemi yapabiliriz.1 degeri axis'te y kordinati gibi dusunelim.inplace default olarak false olarak tanimlidir.
df.drop("3.",0, inplace = True) #Satir silmek icin bu islemi yapiyoruz.0 degeri axis'te x kordinati gibi dusunelim.
'''
NOT:
Fakat sunu unutmayalim sadece tablodan silmek icin bunu yaptik.Onceden yaptigimiz Total satiri ve Avg stunu yine ayni degerde kalacaktir.
Veri yapilarinda da gordugumuz bir kavram, inplace değilse  gibi hafizayi harici ayri bir blokta tutuyor.Eger bir algoritma inplace ise bunu kullanilan hafiza blogunda yapiyor.Bu yuzden inplace false ise  tablo degismiyecektir.
Ornek olarak insertion sort,quick sort bakabilrsiniz bu siralama algoritmasi inplace bir yapidadır.Merge sort, counting sort inplace olmayan siralama algoritmalaridir.Daha bircok ornekte inceleyip farkini anlayabilirsiniz.
'''
df["Index"] = [0, 1, 2, 3]
df.set_index("Index", inplace = True) #index stununu degistiriyor ve stun basligi bu ornekge gore Index olarak yaziliyor.
df.index.names #Index stunun baslik ismini gormek icin bu sekilde yapabiliriz.
df.index.names = ["Konum"] #Default olarak None degerdedir tabloda bir index basligi yazmaz.BU islemlede yine index stununa baslik ismi verebiliriz.
booldf = df > 50 #Bu islemle kontrol yapiliyor tum tablodaki verilerde bu sonucu saglayanlar tabloda true saglamayanlar false olarak yazilip tablo olarak gosteriliyor.
df[df > 50] #Bu kosulu saglamayanlar Nan veya nan gosterecek.
df[booldf] #Bu sekilde yazim yine yukaridaki gibi ayni sonucu verecktir.
df["Avg"] > 50 #Bu sekilde ortalama kontrolu yapiyor yine tablo seklide sonucu cikariyor.
boolavg =  df["Avg"] < 50
df[boolavg] 
df[df["Avg"] < 50] #Bu yukaridaki ile ayni islemler gosterim farkli.Bu iki islem birini yaparsaniz kosulu saglayanlar tabloda gozukecek saglamayanlar tabloda yer almiyacak.
df[(df > 50) & (df < 80)] #Bu sekilde bu sekilde and operatoru ile karsilastirilmis gibi yapacak pandas and kabul etmedigi icin bu sekilde yapiyoruz.
df[(df > 50) | (df < 80)] #Bu sekilde or operatoru ile karsilastirilmis gibi yapacak pandas and kabul etmedigi icin bu sekilde yapiyoruz.
df[(df > 50) ^ (df < 80)] #Bu sekilde xor yapabiliriz.
df[ (df["Avg"] > 50) & (df["Avg"] < 80)] #Yİne benzer filtreleme islemleri yapilabilir.
outerIndex = ["Group1", "Group1", "Group1", "Group2", "Group2", "Group2"]  #Bunlar dis grup icin bu sekilde
innerIndex = ["1.","2.","3.","1.","2.","3."] #Icteki grup icin bu sekilde yaziyoruz yazdirilinca nasil gozuktugu anlasilacaktir.
oii = list(zip(outerIndex,innerIndex))
df = pd.DataFrame([(11,5,6),(7,8,9),(4,5,6),(11,50,60),(70,80,90),(40,50,60)],oii,["Col1", "Col2", "Col3"]) #index bolumune oii ekledik sadece digerleri kullandigimiz yontemler.
print(df["Col2"]) #Yine kolonlari yazdimak icin bu sefer tabloda grup ve icteki index degerleride ekrana gelecek bu islemleri klasorlemek gibi dusunebilirz.
print(df.iloc[0]) #Ilk grup1deki ilk satiri gosteriyor.
print(df.iloc[0][0]) #Ilk grubun ilk satirin ilk stun degeri gostermek icin.
print(df.loc["Group2"].loc["2."]["Col1"]) #Yukardaki islemin farkli gosterimi bunuda kullanabilirsin.
print(df.loc["Group2"]) #Group2 sadece gostermek icin bu sekilde yapabiliriz.
print(df.loc["Group2"].loc["2."]) #Group2 nin 2. satirindaki gostermek icin bu sekilde yapabiliriz.
df.index.names = ["Parti","Konum"] #Yine baslik ekleyebiliriz.
df.xs("Group1") #Bu da baska bir group gostermek icin kullanabiliriz.
print(df.xs("Group1").xs("3.")) #Group1 3. satirini gostermek icin kullanabiliriz.
print(df.xs("Group1").xs("3.").xs("Col2")) #Group1 3. satirini gostermek icin Col2 degeri gostermek icin kullanabiliriz.
print(df.xs("2.",level = "Konum")) #Bu sorguda 2. index degerine sahip satirlari grouplarla birlikte gostermek icin yapiyoruz.level yazmazsak hata veriyor.Default olarak None degerde ve erisim icin level belirtmemiz gerekiyor.
np.nan #as np yapmistik numpy kutuphanesini bu sekilde veriyi NaN veya nan olarak tanimlayabiliriz.
df.dropna() #axis degeri default olarak 0 yani x kordinatina gore bakicak.Satirlarda 1 tane NaN varsa o satiri cikaracak.
df.dropna(1,inplace = True) #NaN olanlari silmek icin kullanabilirsin default yine inplace = 0 olarak tanimlidir.
df.dropna(0,thresh = 2,inplace = True) #Girilen thresh degeri kadar tanimli bir sayi varsa onlari kabul ediyor.Eger NaN sayisi fazla ise bu sefer sayi tanimlari az kaldigi icin ilgili axis ile silme islemi yapiyor.
df.fillna(0) #NaN olarak tabloda gosterilen degerin degerini parantez icinde girdigimizle degistiriyor.
df.sum() #Bu islemle stunlardaki tum verileri topluyoruz.Yine tablo olarak gosteriyor.df.sum(0) bu sekilde yapabilir cunku default olarak axis = None tanimlidir.
df.sum(1) #Bu islemle satir satir tum verileri topluyoruz.
df.sum().sum() #Toplanilan stunlarinda toplamini istersek bu sekilde yaziyoruz kisacasi tablodaki tum verileri topluyoruz.Birinci sum 1 veya 0 olmasi sonucu degistirmez.Sirasiyla sum alabilecak degerler ikili gosterim olarak (0,0),(,),(0,),(1,),(,0)(1,0) eger bu ikili disinda bir degerler verirseniz hata alirsiniz.
df.size #Tabloda kac tane veri oldugunu dondurur.NaN olalarda sayilir fillna yapilsin veya yapilmasin.
df.isnull() #Tabloda NaN olanlari true veya false olarak gosterir.
df.isnull().sum(0) #Stunlardaki NaN eleman sayisini toplar gibi dusunebiliriz veya NaN veri sayisini gosterir kolon bazli.
df.isnull().sum(1) #Satirdaki NaN eleman sayisini toplar gibi dusunebiliriz veya NaN veri sayisini gosterir satir bazli.
df.isnull().sum().sum() #Toplamda kac tani NaN verisi varsa onu gosterir.Sirasiyla sum alabilecak degerler ikili gosterim olarak (0,0),(,),(0,),(1,),(,0)(1,0) eger bu ikili disinda bir degerler verirseniz hata alirsiniz.

dataset = {"Alan":["IT","İnsan Kaynaklari","Pazarlama","Pazarlama","IT","IT"],"Calisan": ["Mustafa","Ali","Kenan","Zeynep","Murat","Ahmet"],"Maas":[3000,3500,2500,4500,4000,2000]} #Bir veri seti olusturduk.
df = pd.DataFrame(dataset) #Veri setini DataFrame yaptik.
subGroup =  df.groupby("Alan") #Burda group olusturduk.axis = 0 default olarak tanimlidir.
subGroup.sum() #Olusturdugumuz gruplarda ayni olanlar toplanacak bu ornekte toplam ilgili alanin maas toplami tablo olacak.
subGroup.count() #Grouplanan verilerin sayisi veriyor bu ornekte  kac kisi ayni alanda  calisiyor gorebiliyoruz.
subGroup.max() #Ilgili grouptaki en buyuk degerler gosterilecek.
subGroup.min() #Ilgili grouptaki en kucuk degerler gosterilecek.
subGroup.mean() #Ilgili grouptaki ortalamasini  gosterilecek.
#subGroup.mean().loc["IT"] yukarida satir ve stundaki ilgili blogun alinmasi islemleri burdada yine gecerli tekrar ele alinmadi.
subGroup.mean()["Maas"]["IT"] #Degeri tablodan bagimsiz olarak almak ve kullanmak icin bu sekilde yapilabilir.
dataset1 = {"Alan":["IT","İnsan Kaynaklari","Pazarlama","Pazarlama","IT","IT"],"Calisan": ["Mustafa","Ali","Kenan","Zeynep","Murat","Ahmet"],"Maas":[3000,3500,2500,4500,4000,2000]}
dataset2 = {"Alan":["IT","İnsan Kaynaklari","Pazarlama","Pazarlama","IT","IT"],"Calisan": ["Mahmut","Ali","Furkan","Sila","Murat","Ahmet"],"Maas":[3000,3500,5500,4500,4000,8000]}
df1 = pd.DataFrame(dataset1)
df2 = pd.DataFrame(dataset2)
df3 = pd.concat([df1,df2],0) #concat ile DataFrameleri birlestiriyoruz.axis = 0 default tanimlidir bu listeyi satir olarak ekliyecegini belirtiyor.
df4 = pd.concat([df1,df2],1) #axis = 1 ornekte esitledik.Bu islemle  stun olarak ekleme yapmasini istedigimzi belirtiyoruz.
dataset1 = {"A":["A1","A2","A3","A4"],"B": ["B1","B2","B3","A4"]}
dataset2 = {"X":["X1","X2","X3"],"Y": ["Y1","Y2","Y3"]}
df1 = pd.DataFrame(dataset1)
df2 = pd.DataFrame(dataset2)
df1.join(df2)#Join islemi kumelerdeki A-B ˅ A˄B  gibi dusunebiliriz.Default degerlri how = left bu hangi kumeyi alacagini belirliyor ve sort = False istersek siralayabilirizde.
df3 = pd.merge(df1,df2, on = "B")#merge() islemi A˄B aliyor.Yani ortak olanlar alniyor.Default how = 'inner' inner join kullaniliyor fakat farkli yontemlerde var.Left join,right join,outer join ... var.on esitlik kontrolu yapacagimiz hangi alansa onu yaziyoruz.
df1.head(3) #dead() girilen deger kadar satir basindan baslayarak almamiza yariyor.Default n = 5 yani sadece 5 satir aliyor.
df1["A"].unique() #Adindan anlasilacagi gibi essiz benzersiz olanlari almamiza yariyor.
df1["A"].nunique() #unique olanlarin sayisi icin nunique() kullaniyoruz.
df1["A"].value_counts() #Belirtilen alandaki degerlerin sayisini gosterir.Bunlarda tek stunlarda degil satirda da bu islemleri yapabiliriz.
def karesi(x):
    return x**2
df["Col1"].apply(karesi) #apply() ile bir fonksiyonu calistirabiliyorum DataFrame uzerinde tum fonksiyonlari kullanabiliriz tek fark func() yerine func yaziyoruz.
kare = lambda x : x**2
df["Col2"].apply(kare) #Lambda fonksiyonlarida kullanabiliriz.
df["Col3"].apply(lambda x : x**2) #Yukaridaki iki batir yerine bu sekilde tek satirda tum islemleri yapabiliriz.
print(df.columns) #Tum kolonlarin isimlerini yazdirir.
print(df.index) #Tum satirin index isimlerini yazdirir.
print(len(df.index)) #Tum satirlarin index sayisini yazdirir.
print(len(df.columns)) #Tum stunlarin sayisini yazdirir.
df.sort_values("Col1",0) #Stunlari siralayabiliriz.axis = 0 defailt olarak belirlenmis 0 yazmayabiriz ama fark anlasilmasi icin bu sekilde yazdim.
df.sort_values("1.",1) #Satiri bu sekilde siralayabilir.Default degereri ascending=True bu kucukten buyuge siralama yapmak icin,inplace=False,kind='quicksort'  siralama alforitmasini degistirebiliyoruz.
df.sort_values("1.",1,False) #ascending false yapiyoruz ve buyukten kucuge siralanmasini sagliyoruz.
df.sort_values("1.",1,"insertionsort") #Burda default olarak quicksort vardi bunu biz ayni zamanda stable olsun dedik ve insertionsort algoritmasi kullanarak yapmasi icin bu islemi yaptik.
df = pd.DataFrame({"Ay" : ["Eylul","Temmuz","Temmuz","Ocak","Ocak","Temmuz","Eylul","Ocak","Eylul"],"Sehir":["Konya","Konya","Konya","Istanbul","Istanbul","Istanbul","Ordu","Ordu","Ordu"],"Nem":[10,25,50,21,47,60,30,80,75]})
df.pivot_table(index = "Ay", columns = "Sehir", values = "Nem")
dataset = pd.read_csv("dosya_adi.csv") #csv dosyalarin okumak icin bu sekilde yapiyoruz.
changed.to_csv("dosya_adi.csv") #Kendimiz bir csv dosyasi olusturmak icin bu sekilde yapiyoruz.Degisiklik yapmis olabiliriz ilgili DataFrame to_csv ile csv  dosyasina donusturuyoruz.
changed.to_csv("dosya_adi.csv", index = False) #index degerlerini almadan kaydetmek icin yapiyoruz.
dataset = pd.read_html("url") #Internet sayfasindaki bir tabloyu bu sekilde okuyabiliriz.
dataset = pd.read_html("url", header = 0) #Tablonun basligini bir tablo degeri olarak alabiliyor bu sekilde basligi ilk satir yapip daha dogru bir okuma yapabilirsiniz.

dosya_ok = pd.read_fileFormat("dosya_adi.fileFormat") #Sadece csv degil bircok dosya formatinida okuyabiliyoruz.
dosya_ok.to_fileFormat("dosya_adi.fileFormat") #Yine ayni sekilde okudugumuz gibi bircok yosya formatiyla kaydetmek mumkun.
#Fakat fileFormat yerine ozel bir yazim olabilir arastirmanizda yarar var fakat genel gosterim olmasi adina bu sekilde yazildi.

#Matplotlib ile Veri Gorsellestirme
import matplotlib.pyplot as plt #Matplotlib kutuphanesini import etmek icin.
%matplotlib inline #jupyter notebook kullandigimizda veri gorsellestirmeyi gormek icin ekelememiz gerekiyor.
plt.show() #Dosya kullandigimizda veri gorsellestirmeyi gormek icin ekelememiz gerekiyor.
x = [0,1,2,3,4]
y = [0,1,4,9,16]
plt.plot(x,y,"blue") #plot() cizme islemi icin ornekte x ve y olarak verdigimiz kordinatlarda cizme islemi yapacak.Cizimdeki renk ornekte blue ama istedgimiz bir renk verebiliriz.RGB kodu olarak girebiliriz.
plt.show() #Jupyter notebook bunu eklemesek calisiyor fakat farkli dosyadan okuyan olabilir icin
plt.subplot(2,2,1) #satir,stun.index degeri olarak veriyoruz bu sekilde 4 tane grafik cizebiliyoruz.
plt.plot(x,y,"black")
plt.subplot(2,2,2) #satir ve stun degerleri buyudukce grafik dahada kuculuyor.2 den kucuk satir ve stun degerlerde ondalik sayilarda hata veriyor.
plt.plot(x,y,"blue")
plt.subplot(2,2,3)
plt.plot(x,y,"green")
plt.subplot(2,2,4) 
plt.plot(x,y,"orange")
plt.show()

fig = plt.figure() #Bir figure olusturmak icin yapiyoruz.
fig = plt.figure(figsize = (5,5)) #figsize ile grafigin boyutunu belirleyebiliriz.
axes = fig.add_axes([0.1,0.1,0.5,0.5]) #Bir grafik eklemek icin yapiyoruz.
#      soldan ne kadar icired olacak,asagidan ne kadar iceride olacak,x duzleminde ne kadar buyuklugu olacak,y duzleminde ne kadar buyuklugu olacak.
axes.set_xlabel("X Kordinati") #x kordinatinda bilgi yazisi gibi dusunebiliriz.
axes.set_ylabel("Y Kordinati") #y kordinatinda bilgi yazisi gibi dusunebiliriz.
axes.set_title("Ilk Grafik")   #Grafigin ust kisminda tablonun  bilgi yazisi gibi dusunebiliriz.
fig,axes = plt.subplots(2,2) #4 tane grafik olusturmak icin yaptik.
plt.tight_layout() #Bu islem grafiklerin arasinda bosluk olmasini sagliyor.
fig,axes = plt.subplots(2,1) #2 tane grafik cizdirdik.
axes[0].plot(x,y) #1.Grafik icin cizim yaptik
axes[1].plot(x,y) #2.Grafik icin cizim yaptik
for ax in axes:#Yukaridaki iki islemi bir for dongusuyle yapilabilir.
    ax.plot(x,y)
plt.show()
fig,axes = plt.subplots(2,1,figsize = (10,10)) #Bu sekilde de grafik boyutu belirlenebilir.
fig.savefig("fig1.jpg") #Bu grafikleri kaydetmek istersek yine benzer sekilde istedigimiz formatta kaydetmek mumkun.Burda normal dosya uzantilari seklinde yazmamiz yeterlidir.
plt.plot(x2,y2,"#00ff00") #Ayni grafikte birden fazla cizim yapilabilir.
fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0,0,1,1])
axes.plot(x,y,"yellow",lw = 10,label = "Dogrusal Grafik") #lw cizimin kalinligini ayarlamaya yariyor.lw yerine linewidht yazilabilir biz kisa oldugu icin lw kullandik.
axes.plot(x2,y2,"blue",label ="Dikey Grafik") #label ile isim cizime isim veriyoruz legend ile gozukecek.
axes.plot(x2,y2,"blue",lw = 5, ls = "--",label ="Dikey Grafik") #ls cizimin nasil gozukmesini istedigimizi belirtmek icin kullaniyoruz yine ls yerine linestyle kullanabiliriz.":" noktali cizim icin , "--" cizikli cizim icin, "-." hem kesikli hem noktali bir cizim icin
axes.plot(x2,y2,"blue",lw = 5, ls = "--",marker = "o", label ="Dikey Grafik") #Marker adindan anlasilacagi gibi isaretleme yapiyor x kordinatindaki degerleri "o" daire icine aliyor.
axes.plot(x2,y2,"blue",lw = 5,marker = "o",markersize = 8,markerfacecolor = "black", label ="Dikey Grafik") #markersize ile isaretleri cizim boyutunu istedigimiz bir degerde yapabiliriz.markarcolor ile isaretlmeye renk varabiliriz.Ayni renk olduklari icin tam net gozukmuyor.
axes.plot(x2,y2,"blue",lw = 5,marker = "o",markersize = 8,markerfacecolor = "black",markeredgecolor = "green", markeredgewidth = 5,label ="Dikey Grafik") #markeredgecolor ile kenarin renk verebiliriz.markeredgewidth
axes.legend() #Bu grafigin sol ust kosesinde cizimleri renk olarak ve label ismine gore tanitim icin ekliyor.
plt.tight_layout()
plt.show()
axes.set_xlim(0,10) #x kordinatinin hangi degerler kadar olacagini burda giriyoruz.Bunlari bir cizim farkli bir degerde basliyorsa grafigi buyutmek icin kullanilabilir.Cizimlerin degrelerinden kucuk verilrse bu seferde o degere kadar cizilecek ve orda bitecek.
axes.set_ylim(0,20) #y kordinatinin hangi degerler kadar olacagini burda giriyoruz.Bu sekilde grafik buyutulurse okuma islemi degisebiliyor noktalarin bazilarinin yerine farkli geliyor ayni oranda bir buyume olmasi icin kullanim nasil olacaksa o sekilde duzenleyebilirsiniz.

#Garbage Collector(gc)
gc.enable() #Otomatik gc calismasini etkinlestirmek icin 
gc.disable() #Otomatik gc calismasini devre disi icin bazi özel manuel  islemler icin otomatik secenek kapatilabilir.
gc.get_objects(generation=None) #Toplayıcı tarafından izlenen tüm nesnelerin bir listesini döndürür, döndürülen liste hariçtir. Generation None değilse, yalnızca o nesildeki toplayıcı tarafından izlenen nesneleri döndür.
gc.is_tracked(obj) #Nesne şu anda çöp toplayıcı tarafından izleniyorsa True, aksi takdirde False döndürür.
gc.is_finalized (obj) #Verilen nesne çöp toplayıcı tarafından sonlandırılmışsa True döndürür, aksi takdirde False.
gc.get_stats()
gc.garbage
gc.DEBUG_STATS #Toplama sırasında istatistikleri yazdırın. Bu bilgi, toplama sıklığını ayarlarken faydalı olabilir.
gc.DEBUG_COLLECTABLE
gc.DEBUG_UNCOLLECTABLE
gc.DEBUG_LEAK

#hashlib
import hashlib
hashlib.new(name, [data, ]*, usedforsecurity=True) #Ornek  h = hashlib.new('ripemd160') 
hash.digest_size #Ortaya çıkan karmanın bayt cinsinden boyutu.
hash.block_size #Karma algoritmanın bayt cinsinden dahili blok boyutu.
hash.name #Bu hash'in kanonik adı, her zaman küçük harflidir ve bu türden başka bir hash oluşturmak için new () için bir parametre olarak her zaman uygundur.
hash.update(data) #Hash nesnesini bayt benzeri nesneyle güncelleyin. Tekrarlanan çağrılar, tüm bağımsız değişkenlerin birleştirildiği tek bir çağrıya eşdeğerdir: m.update (a); m.update (b), m.update (a + b) 'ye eşdeğerdir.
hash.digest() #Şimdiye kadar update () yöntemine iletilen verilerin özetini döndür. Bu, Digest_size boyutunda bir bayt nesnesidir ve 0 ile 255 aralığındaki tüm baytları içerebilir.
hash.hexdigest() #Digest () gibi, tek fark, yalnızca onaltılık rakamları içeren, çift uzunlukta bir dize nesnesi olarak döndürülür. Bu, e-posta veya diğer ikili olmayan ortamlarda değeri güvenli bir şekilde değiştirmek için kullanılabilir.Daha cok bu yontem kullanilir.
hash.copy() #Karma nesnenin bir kopyasını ("clone") döndürür. Bu, ortak bir ilk alt dizeyi paylaşan verilerin özetlerini verimli bir şekilde hesaplamak için kullanılabilir.
hashlib.pbkdf2_hmac(hash_name, password, salt, iterations, dklen=None) #Ornek dk = hashlib.pbkdf2_hmac('sha256', b'password', b'salt', 100000)
hashlib.scrypt(password, *, salt, n, r, p, maxmem=0, dklen=64)
hashlib.algorithms_available #Python da haslib kutuphanesinin destekledigi algoritmalari gosterir.
hashlib.algorithms_guaranteed #Garanti verilen hash islemi algoritmalarini listeler md5 aciklar nedeniyle yukarida listelensede bu listede yoktur.

h = hashlib.sha256()
h.update(b"Bu sifreli bir mesajdir.")
h.hexdigest()

#Flask
pip install flask #Flask framework kurmak icin.
from flask import * #Flask framework import etmek icin
app = Flask(__name__) #
@app.route("/")
def hello():
    return "Hello World"
@app.route("/test") #Bu sekilde alt domainler verebiliriz.
def runTest(): #Hemen altinda bir fonksiyon olmali
    return "Flask run"
if __name__ == "__main__" : 
    app.run(debug = True) #Hatalari gormek icin debug = True yaptik.
python Test.py #Terminalden calistirilmasi daha dogru pyhton shellde url gostemiyor.Serveri kapatmak icin Ctrl + C basarak kapatiyoruz.
#Server acik oldugu surece yapilan degisikleri yenileyerek hemen gorebilirsin her seferinde tekrar serveri acman gerekmiyor.
@app.route("/")
def hello(): #Jinja template ile html kodlarini calistirabiliriz.
    return render_template("index.html") #Dizine templates isimli klasore html dosyalari ekleyip bunu render_template ile kullanabiliriz.

@app.route("/goster")
def shw():
    sayi = 5
    return render_template("index.html", say = sayi) #Degiskeni html sayfasina gondermek icin bu sekilde yapabiliriz.
{{ say }} #HTML saydasinda python kodlarini calistirmak icin {{}} kullanilir. say gondermistik cikti 5 olarak gozukecek.
{% extends "x.html" %}  #HTML saydasina ekliyoruz.Bu islem bir x sayfasindaki Html kodlarini miras almaya yariyor.
{%%} #Normalde {{}} kullanilir demistik fakat include, extends gibi islemler icin bu sekilde yapiyoruz.
'''
{%block or %}
    Hello
{%endblock %}

Bu sekilde blok olusturulabilri or yerine istedigimiz bir isim yazilabilir.
Miras aldigimiz sayfa bu bloktaki islemleri eger degisiklik yapilmak istenirse override ediliyor.
{%block or %}
    Hello world
{%endblock %}

Bu ustekki miras aldigimiz sayfadaki uygulamasi

Bunun uygulamasina ornek olarak bir degiskenin degerini farkli yapmak istersek diger sayfadi bu sekilde yapabiliriz.
''' 
{%include "includes/y.html"%} #templates klose alt klasor olarak includes ekleyip icine y.html dosyasini olusturup icerigini bu sekilde aktarabiliriz.
#templates klasorun ismi degistiginde hata veriyor fakat includes yerine fakli isim verebilirsiniz.
#Klasor olmassa templates veya includes bulamiyor hata veriyor o yuzden klasorlerin icinde olmasi gerekiyor.
return redirect(url_for("hello")) #Sayfaya yonlendirmek icin fakat .html olarakdegil app.route sonra yazdigimiz fonksiyon isimini yaziyor hello yerine
#HTML sayfamizda if else kontrolleri yapabiliriz.
{% if say  == 0 %}
	Basla

(% elif say  == 5 %}
	Boom
{% else %}
{{say}}
{%endif%}

#HTML sayfamizda donguleri kontrolleri yapabiliriz.
{% for i in range(11)%}
	{{i}}
{%endfor%}


@app.route("/goster/<string : id>") #Hata veriyor fakat bu sekilde url ekledigimizi alabiliyoruz.
def shw(id):
    return "ID = " + id  

flash('NP', 'success') #Burda flashta gozukecek mesaji ve kategoruyi yaziyoruz.Flash alert benzeri bir sistem sayfa yenilendiginde gidiyor.
{% with messages = get_flashed_messages(with_categories=true) %}
  {% if messages %}
    {% for category, message in messages %}
		<div class="alert alert-success" role="alert"> #Burda success olan yeri bir degiskenle farkli katagoruler ekleyebilir.
			Basarili #Mesajida yazilan mesaja gore ayarlayabiliriz bununlarin ornegini simdi yapmadim.
		</div> #Birde sayfa yonlendirmelerde ise yariyor bu flash islemi.
    {% endfor %}
  {% endif %}
{% endwith %}

#Django
#!!!!  Bolum 21 Django ilk 4  izlendi geri kalan 42 video izlenmemisti !!!!
pip install Django #Django kurmak icin
pip install Django==3.2.4 #Bu sekilde istediginiz surumu kurabilirsiniz yukaridakinde en guncel surum yuklenecektir.
mkvirtualenv my_django_environment
python -m venv sanalortam 
django-admin startproject mytestsite #Yeni bir django projesi olusturmak icin.
python3 manage.py runserver #Django kendisi dosya olusturuyor ve icindeki manage.py dosyasini runserver komutu ile calistirdigimizda server calisiyor.
python manage.py startapp ilkUygulama #Uygulama olusturmak icin verdigimiz ad onemli bunu ekliyecegiz.
#Daha sonra django olusturdugu klasorde settings.py installed app kismina ilkUygulama ekleyip sonuna virgul ekliyoruz.
#Settings dosyasinda dil ve saat ayari gibi ayarlarida yapilandirabilirsin.
#url.py kenmiz alt domainler ekliyebiliyoruz.
python manage.py migrate #Veri tabani olusturmak icin
python manage.py createsuperuser #Django admin panelinde gecerli  superuser tanimlamak icin createsuperuser kullaniyoruz.
python manage.py makemigrations x #Veri tabaninda tablo olusturmak icin makemigrations kullaniyoruz fakat ayrintili bakmadim.Bu islemden sonra
yine migrate kullaniyoruz.

#Selenium
pip install selenium
#geckodriver isimli browserlara ozel bir driver yuklenmesi gerekiyor.
#Yukaridaki nedenden dolayi guvenlik riski olabileceginden kullanilmamistir.

#Scrapy
pip3 install virtualenv
virtualenv Scrapy #Sanal ortam olusturmak icin biz Scrapy olarak olusturduk baska isimlerde verilebilir.
source Scrapy/bin/activate #Sanal ortam calistirmak icin
scrapy startproject proje_adi #Scrapy projesi olusturmak icin
'''
You can start your first spider with:
    cd proje_adi
    scrapy genspider example example.com

'''

#Yeni bir dosyaya alttaki kodlari yapistiriyor.Bu dosya proje_adi/spiders/ klasorunun altina ekliyoruz.
from pathlib import Path

import scrapy


class baru(scrapy.Spider):
    name = "baru" #Uniqe bir isim vermeliyiz cunku bu isimle calistiracagiz.

    def start_requests(self):
        
        urls = [ 'url' ] #Birden fazla url virgulle ayirip ekliyebiliriz.

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = f'baru-{page}.html'
        Path(filename).write_bytes(response.body)
        self.log(f'Saved file {filename}')
        
#Daha kisa olarak asagidaki yapiya benzer yapilabilir dosyanin icerigi        
class baru(scrapy.Spider):
    name = "baru" 

    start_urls = [ 'url' ]
    
    def parse(self, response):
        icerik = response.css('div.icerik').extract_first()

        yield {
            "icerik": icerik
        }
#Bu kisimi aslinda normal dosya acip write komutu ile yazilmasi mumkun.
        filename = f'baru.html'
        Path(filename).write_bytes(response.body)
        self.log(f'Saved file {filename}')

#cd ../ ilk konuma geliyoruz proje_adi'nin  oldugu konuma
scrapy crawl baru #Calistirmak icin verdigimiz isme gore calistirdik.
scrapy crawl baru -o result.json #Yapilan sorguyu json formatinda da kaydedebiliriz.Her calistridiginda sonuna ekleme yapar dosya ismi ayni ise
scrapy shell "url" #Scrapy shell calistirmak icin
'''
Available Scrapy objects:
[s]   scrapy     scrapy module (contains scrapy.Request, scrapy.Selector, etc)
[s]   crawler    <scrapy.crawler.Crawler object at 0x7f9b30cc4310>
[s]   item       {}
[s]   request    <GET url>
[s]   response   <200 url>
[s]   settings   <scrapy.settings.Settings object at 0x7f9b30cc4460>
[s]   spider     <DefaultSpider 'default' at 0x7f9b3082cdc0>
[s] Useful shortcuts:
[s]   fetch(url[, redirect=True]) Fetch URL and update local objects (by default, redirects are followed)
[s]   fetch(req)                  Fetch a scrapy.Request and update local objects 
[s]   shelp()           Shell help (print this help)
[s]   view(response)    View response in a browser

Shell de kullanabilecegimiz bazi komutlar.
'''
response.css('title') #Web sitesinin basligini almak istersek.
response.css('div').extract() #Sadece sorgulanan bilgileri getirmek icin.Kendisi sorguya ek bilgiler ekliyor bu yuzden kullaniyoruz.
response.css('title::text').extract() #Basligin sadece text kismini almak icin
response.css('title::attr(herf)').extract() #Bir ozelligi almak icin
response.css('title::text').extract()[0] #Liste olak dondurdugu icin saf hali icin [0] eklemesi yaptik.
response.css('title::text').extract_first() #Yukaridaki ayni islemi yapiyor.
response.xpath('//div/text()').extract_first() #Yukaridaki islemin farkli xpath ile yapilmis hali
response.xpath('//div').extract_first() #Text olarak almak istemz isek // ile yapiyoruz.
response.css('div.icerik').extract_first() # . ile class ismine gore cagirabiliyoruz.ID kabul etmiyor.
response.css('div.icerik div').extract_first() #Divlerde class icerik olanin icindeki div almak icin.Ici ice gecmisleri bosluk birakip yazip alabiliyoruz.
response.css('div.icerik div div div a').extract_first() #Bu sekilde zincir sekline alt bolumlere inebiliriz.
#Tarayicilarda olan bir ozelli ile bu islem cok kolay Q veya Sag tik Inspect -> Iligili secilen alanda sag tik -> Copy -> ,,
#Yangi yolla almak istersen mevcut css ve xpath seklinde kendisi direkt otomatik olarak veriyor.
FEED_EXPORT_ENCODING = 'utf-8' #settings.py dosyasina eklersek utf-8 karakterleri destekler.



#Multi Processing
from multiprocessing import Process
from time import sleep


def paralel():
    for i in range(0, 10):
        print("from pc1\t", i)
        sleep(1) 

def paralel2():
    for i in range(0,10):
        print("from pc2\t", i)
        sleep(1)

def mainfunc():
    pc1 = Process(target=paralel)
    pc2 = Process(target=paralel2)
    pc1.start()  #processler baslatiliyor
    pc2.start()
    pc1.join() 
    pc2.join() 
    #join ile mainfunc paralel islemler bittikten sonra devam etmesini sagliyor yoksa paralel islemleri siraya aliyor sonra calisyiryor.
    #join yukaridaki gerekce ihtiyac yoksa eklenmeyebilir yani mainfunc baska islem yoksa veya bir baglilik yoksa gerek yok eklemeye.
    
if __name__ == "__main__":
    mainfunc()


#Parametreli methodlarda multiprocessing
from multiprocessing import Process
from time import sleep


def paralel(x):
    print("from pc1\t", x)
    sleep(1) 

def paralel2(x,y):
    print("from pc2\t", x, y,sep=",")
    sleep(1)

def mainfunc():
    pc1 = Process(target=paralel, args=("1")) #args deyip girmek istedigimiz parametreyi ekliyoruz.Fakat int olarak kabul etmiyor.
    pc2 = Process(target=paralel2,args=("1","2")) #Birden fazla parametreyide yine gonderebiliriz.
    #Bir degiskenden alinacak ise f string yapisi kullanilabilir.
    #pc1 = Process(target=paralel, args=(f"{i}")) gibi bir yapida olabilir.
    pc1.start()
    pc2.start()
	
  
if __name__ == "__main__":
    mainfunc()

#Thread
from threading import Thread
from time import sleep


def paralel():
    for i in range(0, 10):
        print("from th1\t", i)
        sleep(1) 

def paralel2():
    for i in range(0,10):
        print("from th2\t", i)
        sleep(1)

def mainfunc():
    th1 = Thread(target=paralel)
    th2 = Thread(target=paralel2)
    #Threadlerde yine args ayni yapi sekilde kullanabilirsin parametreli kullanilabiliyor.
    th1.start()
    th2.start()
    th1.join()
    th2.join()

    
if __name__ == "__main__":
    mainfunc()

 
#PyScript
<link rel="stylesheet" href="https://pyscript.net/latest/pyscript.css" />
<script defer src="https://pyscript.net/latest/pyscript.js"></script> 
#Yukaridaki iki satiri head etiketinde tanimlamamiz gerekiyor.
<py-script> print('Now you can!') </py-script> #Python kodlarini <py-script></py-script> tag arasinda tanimliyoruz.
#py-script taglarini body taginin altinda calistiriyoruz.
<py-repl id="my-repl" auto-generate=true> </py-repl> #Python calistirilabilir code editor acmak icin.

#Gerekli kutuphaneleri eklemek icin.
<py-config> 
    packages = ["matplotlib", "pandas", "numpy","scikit-learn", "requests", "pyodide-http"]
</py-config>
#Desteklenen paketler : https://pyodide.org/en/stable/usage/packages-in-pyodide.html

"""
En cok kullananlari ve bildiklerimi ekledim secebilirsin.
Paket ve kutuphane sayisi arttikca yuklenme suresi artiyor bu yuzden ihtiyacin olanlari eklemen iyi olur veya bekleyebilirsin.

packages = [    "matplotlib", "pandas", "numpy","scikit-learn", "pytest", "pytest-benchmark",
                "scipy", "xgboost",  "biopython", "beautifulsoup4", "cryptography", "opencv-python"

           ]
"""

#Buton ekleyip bastiginda calistirmak icin fonksiyon tanimlayip yapmak istedigimiz islermeleri uygulayabiliriz.
<button py-click="clicked()" id="get-tikla" type="button" class="btn btn-primary">Tikla</button> #bootstrap ozelliklerini kullan.
<py-script> 
      def clicked():
        print("Hello World")
</py-script>

<py-script>
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyodide.http import open_url

url = "https://raw.githubusercontent.com/jokecamp/FootballData/master/Turkey/SuperLig/2015-2016/superlig.json"
veri = pd.read_json(open_url(url)) #Fakat bazi url'den veriyi alamiyor.
print(veri.tail())
</py-script>


<py-config> 
    packages = ["matplotlib", "pandas", "numpy","scikit-learn"]

    [[fetch]]
    files = ["./example.py"]
</py-config>

python3 -m http.server #Dosya Server uzerinde acmak gerekiyor.Diger turlu acmiyor.
http://localhost:8000/ #Localhost aciyorsun. Daha sonra ilgili html dosyaya tikla.
#Sayfada artik dosya ile ilgili islemler yapabilirsin.


<div class="mb-3">
    <label for="formFile" class="form-label">Dosya Sec</label>
    <input class="form-control" type="file" id="File" accept=".py, .pyi, .pyc, .pyd, .pyw, .pyz, .pyo, .txt, .ipynb, .pdf">
</div>

<py-script>
    import js
    from pyodide.ffi.wrappers import add_event_listener

    async def calistir(e):
        dosyalar = e.target.files
        dosya = dosyalar.item(0)

        kod = await dosya.text()
        exec(kod)

    get_dosya = js.document.getElementById("File")
    add_event_listener(get_dosya, "change", calistir);
</py-script>

#print(kod) deyip dosyadaki kodlari yazdirabilirsin.Dosya configde tanimli degilse
#Bu yontemle alinan dosya config'e eklenmden calisabiliyor.
#Fakat repl calistiramiyorsun onun icin halen repl eklemen lazim.Kodu kopyalayip calistirmalisin o sekilde calisir yine


#Code degisiklik yapilmali grafik gosterimi icin simdilik bu sekilde birakiyorum.Hata vermiyor ama gostermiyorda.
<!-- <py-script output="lineplot" src= "file.py"> --> 
#id ile output ayni olmasi gerekiyor.
<div id="plot"></div> 
<py-script output="plot">
  import matplotlib.pyplot as plt
  
  fig, ax = plt.subplots()

  year_1 = [2016, 2017, 2018, 2019, 2020, 2021]
  population_1 = [42, 43, 45, 47, 48, 50]
  
  year_2 = [2016, 2017, 2018, 2019, 2020, 2021]
  population_2 = [43, 43, 44, 44, 45, 45]
  
  plt.plot(year_1, population_1, marker='o', linestyle='--', color='g', label='Country 1')
  plt.plot(year_2, population_2,  marker='d', linestyle='-', color='r', label='Country 2')
  
  plt.xlabel('Year')
  plt.ylabel('Population (M)')
  plt.title('Year vs Population')
  plt.legend(loc='lower right')
  
  fig
</py-script>

#Streamlit
pip3 install streamlit
streamlit run file.py #Python dosyasini calistirmak icin.
streamlit run https://raw.githubusercontent.com/... #Github gists'ten de
streamlit cache clear #Cache temizlemek icin Ctr+C sonra terminale yaz.
streamlit config show #Config ayarlarina bakmak icin.
streamlit docs #Dokuman gormek icin kendi sitesindeki sayfayi aciyor browserdan.
streamlit version #Versiyonu gormek icin.

nano ~/.streamlit/config.toml
[browser]
gatherUsageStats = false

#Config dosyasi yoksa da kaydettikten sonra ayar degisecektir.Yukaridaki ayar telemetry verilerini gondermemek icin.
[server]
sslCertFile = '/path/to/certchain.pem'
sslKeyFile = '/path/to/private.key'
#HTTPS icin ssl sertifikalarini tanimlaman gerekiyor.
[theme]
base = "dark"
#Surekli tema ayarini degistirmek istemiyorsan.


st.set_page_config(
    page_title="Streamlit",
    page_icon="url",
)
#Sayfanin title ve icon ekleme islemi yapabilirsin.Icon icin url ekliyebilirsin.

import streamlit as st
st.write("Hello World") #Ekrana yazi yazdirmak icin.

t = st.text('Loading data...')
time.sleep(3)
t.text("Done....") #Yaiyi degistmek icin. write hata veriyor bu yontem o yuzden text ile yapiyoruz.

with st.echo(): #Blogun altindaki herseyi yazdirmak icin.
    ...


st.help(pandas.DataFrame) #Help bolumunu yazdirabilirsin.
st.help(my_func) #Kendin olusturdugun fonksiyonlarinda help gosterebilirsin.

#Farkli yazi formatlari
st.title('Title') #Baslik h1 vermek icin.
st.subheader('Raw data') #Baslik h2 vermek icin.
st.caption('This.') #Kucuk fontta yazi yazmka icin.
st.markdown("# Main page") #Markdown yazi formati ile yazabilirsin.
st.divider() #Cizgi line ile bolmek icin hr tag gibis
st.sidebar.markdown("# Main page") #Yan menude gostermesi icin.

st.json({
    'foo': 'bar',
    'baz': 'boz',
    'stuff': [
        'stuff 1',
        'stuff 2',
        'stuff 3',
        'stuff 5',
    ],
})
#Json verisini gostermek icin.

code = '''print("Hello World")'''
st.code(code, language='python') #Kodlari renklendirilmis sekilde yazar.Programlama dillerinin isimi kucuk harfle yazilmalidir.
st.latex(r'''a + ar + a r^2''') #Formulleri rahat gostermek icin.

st.metric(label="Temperature", value="70 °F", delta="1.2 °F") #Metric ile degerleri sekilli  gosterir.
st.table(veri) #Aralarinda bosluk birakarak tablolastirir.
st.dataframe(veri) #Normal DataFrame gosterir.DataFrame header renkli table renksiz.


with st.expander("See Answer"): #Acilir pencerede gostermek icin.
    code = '''
    def karesi(n):
        return n**2
    '''
    st.code(code, language='python')
    
    
st.error('This is an error') #Hata gostermek icin.
st.warning("This is a warning") #Farkli renklendirme icin.
st.info('This is a purely informational message')
st.success('This is a success message!')
#icon parametresi ile icon da ekliyebilirsiniz hepsine.
e = RuntimeError('This is an exception of type RuntimeError')
st.exception(e) #Hata gostermek icin

#Grafiksel gosterim
st.line_chart(chart_data) #Cizgi Grafigi olusturmak icin.
st.bar_chart(hist_values) #Cubuk grafigi icin.
st.map(map_data) #Veriyi haritaya aktarmak icin kullaniliyor.
bar = st.progress(0)
bar.progress(i + 1)
st.pyplot(fig)
#Folium 3rd part olarak kullanabiliyorsun.
#Daha fazla grafiksel gosterim icin : https://docs.streamlit.io/library/api-reference/charts

#Menu veya sekme seklinde gecis yaprak gostermek icin.
tab1, tab2 = st.tabs(["Chart", "Data"])
data = np.random.randn(10, 1)

tab1.subheader("A tab with a chart")
tab1.line_chart(data)

tab2.subheader("A tab with the data")
tab2.write(data)

#Input
isi =  st.slider('Sicaklik')  #Slider olusturmak icin.Liste de verilebiliyor icinden seciyor.
st.metric(label="Temperature", value=f"{isi} °C") #Default olarak 0-100
tip =  st.slider('Bahsis', min_value=20, max_value=120)  #max ve min degerleri verebilirsin.
st.write(f"Odenen tip = {tip}")

name = st.text_input("Isminiz") #Input text alani eklemek icin
st.write(f"Hello, {name}")

st.text_input("Your name", key="name")
isim =st.session_state.name #Degerini id olarak da dusunebilirsin.
st.write(f"Hello, {isim}")

txt = st.text_area("Paragraf") #Text_area vermek icin.
if(txt != ""):
    st.write('Roman:', txt)

puan = st.number_input('Puan', min_value=0,max_value=10) #Sayi girdisi almak icin max, min vererek belli aralikta kalabilir.
st.write('Puan ', puan)

t = st.time_input('Set an alarm for', datetime.time(6, 00)) #Saat ve dk alabilirsin. 15dk olarak artiyor.Girilen saat defaulttur.
st.write('Alarm is set for', t)

d = st.date_input(
    "When\'s your birthday",
    datetime.date(2019, 1, 1)) #Tarih almak icin girilen deger defaulttur.
st.write('Your birthday is:', d)

kabul = st.checkbox('Kabul et') #checkbox eklemek icin.
if kabul:
    st.write('Pisman olacaksin!')



option = st.selectbox(
    'Which one do you like best?',
     veri["City"]
) 
#Selectbox ile verilen listeden birini secmesi saglanabilir.

st.write(option)

add_slider = st.sidebar.selectbox(
    'Select a range of values',
    veri["City"]
)
#sidebar ekledigimizde bu secimleri soldaki menude yapiyoruz.
st.write(add_slider)

start_price, end_price = st.select_slider(
    'Select a range of price',
    options=[10, 50, 100, 200, 500, 1000, 3000],
    value=(10, 100))
st.write('You selected ', start_price, '-', end_price)
#Araligi almak icin value default bir aralik verebiliriz.

options = st.multiselect(
    'What are your favorite colors',
    ['Green', 'Yellow', 'Red', 'Blue'],
    ['Yellow', 'Green']
)
#Ikinci dizi default secili hangileri gelsin istersek.

st.write('You selected:', options)

chosen = st.radio(
        'Sec',
        ("Evet", "Hayir")
)

if (chosen == "Evet"):
    st.write("Kabul ettin.Dikkat et.")
        
if st.button('Press'):
    st.write('Tiklandi')

left_column, right_column = st.columns(2)
but = left_column.button('Press me!')

if (but):
    st.write("Tiklandi.")
    
    
with st.sidebar: #Bu sekilde altindakilerin hepsi sidebar da gozukecek.
    with st.echo():
        st.write("This code will be printed to the sidebar.")

    with st.spinner("Loading..."): #Yukleme gostermek icin donerli
        time.sleep(5)
    st.success("Done!")
    
    
name = st.text_input('Name')
if not name:
  st.warning('Please input a name.')
  st.stop() #Calismasini durdurmak icin.
st.success('Thank you for inputting a name.')

st.experimental_rerun() #Tekrar calistirilmasi icin.

form = st.form("my_form")
form.slider("Inside the form")
st.slider("Outside the form")

# Now add a submit button to the form:
form.form_submit_button("Submit") #Form element gibi yapabilirsin.
    
#Yuklenme islmelerin gostermek icin
with st.spinner('Wait for it...'):
    time.sleep(5)
st.success('Done!')

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1, text=progress_text)   

text_contents = '''This is some text'''
st.download_button('Download some text', text_contents) #Indirme islemi icin yaziyi kaydetiyor.

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(my_large_df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv',
) 
#csv dosyasi olarak kaydetmek icin.

with open("flower.png", "rb") as file:
    btn = st.download_button(
            label="Download image",
            data=file,
            file_name="flower.png",
            mime="image/png"
) 
#Resimi kaydetmek icin.

#Fotograf cekmek icin.
picture = st.camera_input("Take a picture")

if picture:
    st.image(picture)

color = st.color_picker('Pick A Color', '#00f900') #Renk secmek icin.
st.write('The current color is', color)


from PIL import Image
image = Image.open('sunrise.jpg') #Resim acmak icin.
st.image(image, caption='Sunrise by the mountains')

#Birden fazla dosya acmak icin.
uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)
    st.write(bytes_data)

#Ses dosyasi icin.
audio_file = open('myaudio.ogg', 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/ogg')
sample_rate = 44100  # 44100 samples per second
seconds = 2  # Note duration of 2 seconds
frequency_la = 440  # Our played note will be 440 Hz
# Generate array with seconds*sample_rate steps, ranging between 0 and seconds
t = np.linspace(0, seconds, seconds * sample_rate, False)
# Generate a 440 Hz sine wave
note_la = np.sin(frequency_la * t * 2 * np.pi)
st.audio(note_la, sample_rate=sample_rate)

#Video icin.
video_file = open('myvideo.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

import pydeck as pdk #Grafiksel gosterimleri harita uzerinde yapmak icin.
chart_data = pd.DataFrame(
   np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
   columns=['lat', 'lon'])

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=37.76,
        longitude=-122.4,
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=chart_data,
           get_position='[lon, lat]',
           radius=200,
           elevation_scale=4,
           elevation_range=[0, 1000],
           pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=chart_data,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=200,
        ),
    ],
))

import folium
from streamlit_folium import st_folium
m = folium.Map(location=[39.949610, -75.150282], zoom_start=16) #location enlem ve boylam verip istersek zoomlayabiliriz.
folium.Marker(
    [39.949610, -75.150282], popup="Liberty Bell", tooltip="Liberty Bell"
).add_to(m)
#Marker eklemekte mumkun.
st_data = st_folium(m, width=725) #folium haritasini render etmesi icin.
st_folium(m, width=725, returned_objects=[]) #Hergangi bir veri haritadan dondurmemesi icin ekliyoruz.
output = st_folium(m, width=725, returned_objects=["last_object_clicked"]) #Eger tiklamada geri kordinat dondurmesi icin.
#Listenin icine dirdigimiz deger gozukuyor ve deger dondurmesini sagliyor.
st.write(output) #Kordinatlari gorebiliyoruz.

from folium.plugins import Draw
m = folium.Map(location=[39.949610, -75.150282], zoom_start=5)
Draw(export=True).add_to(m) #Harita uzerinde cizim yapmamiz icin.
output = st_folium(m, width=700, height=500)
st.write(output)

from streamlit_folium import folium_static
m = folium.Map(location=[39.949610, -75.150282], zoom_start=5)
folium_static(m, width=700, height=500)

import folium.plugins
m = folium.plugins.DualMap(location=[39.949610, -75.13], zoom_start=16) #Iki harita yan yana gostermek icin.
tooltip = "Liberty Bell"
folium.Marker(
    [39.949610, -75.150282], popup="Liberty Bell", tooltip=tooltip
).add_to(m)
st_folium(m, width=2000, height=500, returned_objects=[])

#Ornek
veri = pd.read_csv("./kordinat2.csv")
sec = add_slider = st.sidebar.selectbox(
    'Select a city of values',
    veri["il"]
)
secim = veri[veri["il"] == sec]
m = folium.Map(location=secim.iloc[:,2:].values[0], zoom_start=16)
for i in range(81):
    folium.Marker(
        veri.iloc[i:i+1,2:].values[0], popup=veri["il"].values[i]
    ).add_to(m)
st_folium(m, width=725, returned_objects=[])

"""
Tek boyut : Vector, Series
Iki boyut : Matrix
Uc boyut : Tensor
#Tensor aslinda cok katmanli matrix'lere verilen isimdir.Ornegin RGB degerlerinden olusan matrix katmanlari ornek verilebilir.
#Iki katmanladi olsa yine Tensor adini alir yani 1'den fazla katmana sahip ise matrix'ler Tensor dioruz.
DataFrame en tek boyutu olanlara (iki veya daha cok boyutlu olabilir.) adi da verilemektedir.
"""
#Harita gorseli yapmak icin Folium kutuphanesini kullanabilirsin.
#Web gorunum kazandirmak icin https://streamlit.io/ kullanabilirsin.

#Not: np.random.randint?  sonuna ? eklediginizde bilgileri gormek icin kullanabilirsin.Hangi parametreler var.Nasil yaziliyor gibi

#Tabula
pip3 install tabula-py
#PDF to CSV for tables
import tabula
input_path = "https://www.test.pdf" #Bir dosyanin path yada url olarakta verebilirsin.
df = tabula.read_pdf(input_path, pages="all") #DataFrame olarak alip proje icinde kullanmak istersen.
tabula.convert_into(input_path, "output.csv", output_format="csv", pages="all") #CSV haricinde farkli formatlar da destekliyor.
tabula.convert_into(input_path, "output.csv", output_format="csv", pages=10) #pages parametresi ile sayfa belirtebilirsin.
#pages default degeri 1
tabula.convert_into(input_path, "output.csv", output_format="csv", pages=[10,17,21]) #Liste olarak sayfa numaralari verilebilir.
tabula.convert_into(input_path, "output.csv", output_format="csv", pages="77-81") #Belli bir sayfa araligi verebilirsin.
tabula.convert_into(input_path, "output.csv", output_format="csv", pages="77-78,80-81") #virgil ile eklemeler yapabilirsin.
#pages="77-78,80-81" bu yontemle sayfa atliyabilirsin.

import google.colab import files
dosya = files.upload() #Dosya secip colab yuklemnin farkli bir yolu

#Numpy
import numpy as np

list1 = [1,2,3,4,5]
arr = np.asarray(list1) #asarray ile numpy array donusturmek icin.
arr = np.array([1,2,3,4])
print(arr)
print(arr[3]) #Index'teki elemana erismek yine listelerde oldugu gibi
print(arr[[2,3,8]]) #R c(2,3,8) kullanim ile benzer sekilde degerlere erismek mumkun.
print(arr[arr % 2 == 0]) #R oldugu gibi belirli kosulu sagliyanlari yine alabiliyoruz.Yani kosullu sorgular yapilabilir.
print(arr[arr > 5]) #Bu gibi islemlere Boolean Indexing deniyor.
print(arr>5)
print(np.all(arr < 7)) #Hepsinde saglamadiginda True doner.
print(np.any(arr < 7)) #Bir sagladiginda True doner.
np.where(z>1.5)[0] #where ile hangi konumda olduklarini bulabiliriz. [0] almamizin nedeni tuple dondugu icin.

print(arr + arr2)
print(np.add(arr,arr2)) #add + ile ayni islemi yapar.
print(arr - arr2)
print(np.subtract(arr,arr2)) #subtract - ile ayni islemi yapar.
print(arr * arr2)
print(np.multiply(arr,arr2)) #multiply * ile ayni islemi yapar.
print(arr / arr2)
print(np.divide(arr,arr2)) #divide / ile ayni islemi yapar.
#Dort islemi iki array uzeriden gosterdik fakat sayi ile de bu islemler yapilabiliyor.Not : Arraylerin boyutu  ayni olmalidir.
#R yapilabilen bircok islemi destekliyor.
#element-wise iki arrayin her bir elemanlari icin ile yapilan islemlerdir.array-wise tum array ile yapilan islemlerdir.
arr[2] = 10 #Elemani degistirmek icin yine listelerde oldugu gibi
arr = np.append(arr,11) #Sona eleman eklemek icin.
arr = np.append(arr,[20,30]) #Birden fazla degeri de ekliyebilirsin.
arr = np.insert(arr, 1, 11) #Araya eleman eklemek icin.
#              dizi, index, eleman
arr = np.insert(arr, 4, [1,1,2]) #Yine birden fazla eleman ekleyebiliz. 
arr = np.delete(arr, 3) #Verdigimiz indexteki degeri silmek icin.
arr = np.delete(arr, [3,6]) #Index degerlerine gore birden fazlasi da silinebilir.
print(type(arr)) #<class 'numpy.ndarray'>
arr = np.zeros((2,3)) #Tum degerleri 0 olan 2x3 array tanimlamak icin
arr = np.ones((2,3)) #Tum degerleri 1 olan 2x3 array tanimlamak icin
arr = np.ones((2,3)) * 5 #Bu islem ile tum elemanlari 5 ile carpiyoruz.
arr = np.eye(3) #3x3'luk birim matrix olusturyor.1'ler diyagonel sekilde
arr = np.identity(3) #Yukaridaki ile ayni islemi yapiyor.
arr = np.eye(2,3) #Diyagonel olmasada 1'ler yine capraz oluyor ayni uzunlukta olmadigi icin bu sekilde oluyor.
arr = np.eye(3, k= 1) #+ oldugunda ileri kaydiriyor birleri
arr = np.eye(3, k= -1) #- oldugunda asagi kaydiriyor birleri
arr = np.diag([2,5,7,8,9]) #Diyagonel bir matrix olustuyor diyagondaki degerlere listedeki elemanlari yerlestiriyor.
#Listede istediginiz kadar eleman ekleyip cikartabilirsiniz diyagon yapisni bozmadan sirayla capraz sekilde yerlestiriyor.
arr = np.full((2,3), 5) #Tum degerleri 5 olarak dolduruyor.
arr = np.arange(1,11,2) #range oldugu gibi balangic, bitis, adim degerleri vererek dizi olusturmak icin
arr = np.arange(10).reshape(2,5) #Eleman sayisina gore diziyi tekrar sekil olarak duzenleyebiliriz.

arr = np.arange(9).reshape(3,3)
arr = np.append(arr, [[2,3,6]], axis=0) #3x3 oldugundan bu sekilde sekile uygun ekleme yapilmalidir yoksa hata verir.
arr = np.append(arr, [[2],[3],[6]], axis=1) #Stuna eklemek icin bu sekilde
arr = np.insert(arr, 2, [[2,3,6]], axis=0)
arr = np.insert(arr, 2, [[2],[3],[6]], axis=1)

arr = np.delete(arr, 1, axis=0) #Satir silmek icin axis = 0 yapiyoruz.
arr = np.delete(arr, 1, axis=1) #Stun silmek icin axis = 1 yapiyoruz.
arr = np.delete(arr, [2,4], axis=1)

arr1 = np.arange(2,5)
arr2 = np.arange(9).reshape(3,3)
arr = np.vstack((arr1, arr2)) #vstack == vertical stack .arr1 arr2 basina eklemis oluyoruz.
arr = np.hstack((arr1, arr2.reshape(9,))) #hstack == horizontal stack. Yatayda ekleme yapiyor.
#Fakat boyutlari fakli oldugu icin reshape yapmak zorunda kaldik.

print(np.pi) #pi degeri icin np.pi kullanabilirsin.
arr = np.linspace(1,10,3) #Linspace ise baslangic, bitis, kac sayi olacagini belirtir.linspace == linear space
arr = np.linspace(1,10,100)
#Not baslangic ve bitis degerlerini dahil eder.
arr = np.linspace(1,10,10, endpoint=False) #Bitis degerini dahil etmemek icin.
arr = np.random.random((2,3)) #2x3 random degerleri olan matrix olusturmak icin. random ile ayni islemi ve deger araligindadir.
arr = np.random.randint((2,2,2)) #Bu islem ile 2'ye kadar 3 tane int degerde vektor olusturmak icin
arr = np.random.randint(1,10, size=(2,3)) #baslangic , bitis, size ile vektorun kaca kaclik oldugnu belirtmek icin kullaniyoruz.
arr = np.random.randint(1,10, size=(5,)) #Tek boyutlu yapmak istersek.
#Yine deger araligin randint oldugu gibi bitisi dahil etmiyor baslangic var.
arr = np.empty((2,3)) #Bos birakmiyor deger veriyor.
arr.fill(5) #empty ile olsuturdugumuz dizinin tum elemanlari 5 ile dolduruyoruz.
print(arr.ndim) #ndim kac boyutlu oldugunu gosterir.
print(arr.shape) #Dizinin seklinin nasil oldugnu gosterir.Eksendeki eleman sayilari ile gosterir.
print(arr.size) #Dizinin kac elemani oldugunu gosterir.
print(arr.nbytes) #Kac byte oldugunu gosterir.Tipine gore byte degeri degisiyor. int64 int32 farkli degerde donderiyor.
print(arr.dtype) #Dizideki verilerin tipini gosterir. dtype == data type
arr = np.arange(10, dtype= np.int32) #dtype diziyi tanimlarken tipini kendimiz belirtmek istersek kullanabiliriz.
#Fakat coklu tip desteklemiyor.Tek bir tip tanimlayabilirsiniz.Ornek olarak tum elemanlar int olmali int str birlikte kabul etmez.
arr = np.array([1,"2",3,4]) #Hepsini stringe cevirdi int olarak kalmadilar.Bu isleme upcasting deniyor.
arr = np.array([1,2.2,3,4]) #Bu seferde hepsini float yapiyor.
#Bunun neden veri kaybini onlemek tum degerleri bir ust tip cast ediyorki kayip yasanmasin. 
#Ornek int(5.3) 5 cikacak yani 0.3 degeri almiyacak buda kayip demek oluyor.Bu yuzden upcasting yapiyor.
arr = np.array([1.1,2.2,3.5,4.7], dtype= np.int64) #Bu islem ile downcasting yaptik.
arr = arr.astype(np.complex128) #astype ile tip donusumu yapabiliriz.
print(arr.dtype)
"""
Array-protocol type strings
'?'         boolean
'b'         (signed) byte
'B'         unsigned byte
'i'         (signed) integer
'u'         unsigned integer
'f'         floating-point
'c'         complex-floating point
'm'         timedelta
'M'         datetime
'O'         (Python) objects
'S', 'a'    zero-terminated bytes (not recommended)
'U'         Unicode string
'V'         raw data (void)

dtype object:
int             int_
bool            bool_
float           float_
complex         cfloat
bytes           bytes_
str             str_
buffer          void
(all others)    object_
"""
arr = np.array([1,2,3,4])
arr2 = np.array([2.4,3.2,4.4,5.8])
print(arr+arr2)
print((arr+arr2).dtype) #Floata cevirerek toplar.
print(np.sqrt(arr)) #sqrt ile elemanlarin kare kokunu alabiliriz.
arr = np.array([-1,2,3,4])#float olarak tanimladigimiz degerler icin complex cikacak elemanlar varsa onu nan olarak gosterir.
arr = np.array([-1,2,3,4], dtype=np.complex128) #Bu sekilde veri kaybimiz olmaz.

arr = np.array([-1,-2,-3,4], dtype=np.complex128)
arr = np.sqrt(arr)
print(arr)
print(arr.real) #Gercek (real) kismini almak icin
print(arr.imag) #Sanal(imaginary) kismini almak icin
print(arr.imag.dtype) #Real ve imaginary kisimlarin ayri ayri baktimizda float tipinde donderir.
np.save("dosya_adi.npy",arr) #Diziyi dosyaya kaydetmek icin.
arr2 = np.load("dosya_adi.npy") #Dosyadaki diziyi okumak icin.

np.savetxt("dosya_adi.txt",arr) #Diziyi txt dosyasi olarak kaydetmek icin.
arr = np.loadtxt("dosya_adi.txt") #txt dosyasini yuklemek icin.
np.savetxt("dosya_adi.txt",arr, fmt="%.i") #int olarak kaydetmek icin fmt parametresini kullanabilirsin.
#fmt gosterim olarak sadece int degil diger tipler icinde farkli sekillderde kullanabilirsin.

arr = np.arange(10)
arr2 = arr
print(np.shares_memory(arr, arr2)) #Ayni hafizayi paylasip paylasmadigini kontrolu icin True veya False donderir.
arr2 = arr.copy() #Farkli hafizlarda tutulmasi ve birbirlerini etkilememesi icin.Deep copy icin copy() kullaniyoruz.
arr2 = arr.view() #Shallow copy icin view() kullaniyoruz.

print(arr.T) # T ile trasnpose almak icin kullaniyoruz.
print(np.setdiff1d(arr, arr2)) #arr olup arr2 olmayanlari bize gosterecek.
print(np.setdiff1d(arr2, arr)) #Dikkat ayni sonucu vermez arr2 olup arr olmayanlari gosterecek.
print(np.union1d(arr,arr2)) #Birlesim icin.
print(np.in1d(arr,arr2)) #Degeleri icerip icermedigini gosterir. Matrix seklinde True veya False donderir.
print(np.unique(arr)) #Essiz olanlari almak icin.Tekrar edenleri almiyacak set ile yaptigimiz islem gibi
print(np.sort(arr)) #Silamak icin
print(np.sort(arr, axis=0)) #Stun bazli siralar.
print(np.sort(arr, axis=1)) #Satir bazli siralar.
print(np.sort(arr, kind= "quicksort")) #king ile siralama algoritmalari verebiliyoruz.
#Algoritmalar: "quicksort", "heapsort", "mergesort", "timsort"
arr.sort #Bu sekilde siralama yaparsak arr artik sirali olarak degisecek.Digerinde arr ayni sadece cagirdigimizda sirali donuyordu
print(arr == arr2) #Elemanlar sirasi ile karsilikli esitler mi kontrol etmek icin.Matrix seklinde True veya False doner.
print(np.array_equal(arr,arr2)) #Burda diziler birbirne esitmiyi kontrol ederiz.True veya False doner.
print(np.concatenate((arr,arr2))) #Dizileri birlestirmek icin np.concatenate() kullaniyoruz.
print(np.concatenate((arr,arr2))) #Cok  boyutlular icin kolon veya satir bazli birlestirme icin axis parametresi girilmelidir.
print(np.concatenate((arr,arr2), axis=0)) #Satir bazli birlestirme icin.
print(np.concatenate((arr,arr2), axis=1)) #Stun bazli birlestirme icin.
print(np.split(arr,5)) #Kac parcaya bolmek istiyorsak diziyi degeri verip parcaliyor.Veri sayisina gore tam bolecek sekilde olmali
print(np.split(arr,2)) #Satir sayisindan fazla kabul etmiyor cok boyutlu diziler icin.
print(np.array_split(arr,10)) #array_split daha hatasiz sonuclar icin splite gore daha cok kullaniliyor.
#split bazi bolmelerde yetersiz kaliyor.Bunun yerine array_split kullaniliyor.
print(np.resize(arr,(5,2))) #Sekil olarak tekrar duzenlemek icin resize() kullaniyoruz.
print(np.sqrt(arr)) #Karekok almak icin sqrt() kullanabiliriz.
print(np.power(arr,2)) #Us almak icin power() kullanabiliriz.
print(np.exp(arr)) #loge almak icin exp() kullanabiliriz.
print(np.sum(arr)) #Dizinin toplamini almak icin.
print(arr.sum(axis=0)) #Stun toplamini bulmak icin axis = 0 yaptik.
print(arr.sum(axis=1)) #Satir toplamini bulmak icin axis = 1 yaptik.
print(arr.min()) #min degeri bulmak icin.
print(arr.max()) #max degeri bulmak icin.
print(arr.argmin()) #Ilk buldugu min degerin indexini verir.
print(arr.argmax()) #Ilk buldugu max degerin indexini verir.
print(arr.mean()) #Ortalamasini almak icin.
print(arr.mean(axis=0)) #Stunlarin ortalamasini alir.
print(arr.mean(axis=1)) #Satirlarin ortalamasini alir.
print(np.median(arr)) #Ortanca deger veya Q2
print(np.median(arr, axis=0)) #Stunlarin ortanca degerini alir.
print(np.median(arr, axis=1)) #Satirlarin ortanca degerini alir.
print(arr.std()) #Standard deviation == Standart Sapma
print(arr.std(axis=0)) #Stunlarin standart sapmalarini alir.
print(arr.std(axis=1)) #Satirlarin stanart sapmalarini alir.
print(np.abs(-9)) #np.abs ile sayinin mutlak degerini alabiliriz.
veri = veri.replace(np.nan, 0) #Bos olan degeri kendimiz bir deger atiyabiliyoruz.
np.random.seed(11) #R benzer sekilde ayni sonuclari almak icin seed() istedigimiz bir degeri verebilir.
#Ayni degerde calistirildiginda sonuclar ayni cikacaktir.Bu islemi kontrol icin veya baskalarida ayni sonucu alsin die yapiyoruz.
np.random.normal(size=1000) #Normal dagilima uyan rastgele degerler uretmek icin normal kullaniyoruz.
#size parametresini kac tane deger olusturacagini belirmek icin kullaniyoruz.

#Not Numpy ve Pandas and or matik operatorleri yerine & ve | kullanilir.C dili ile yazildiklari icin.

#Pandas
pip3 install --user --upgrade pandas #pandas yerine istedigimiz bir kutuphaneyi yazabiliriz.Guncellemek icin kullaniliyor bu satir
pip3 show pandas #pandas yerine istedigimiz bir kutuphaneyi yazabiliriz.Kutuphane ile alakali bilgiler verir.
#Yuklu degilse hata verir.Yuklu mu degil mi kontrolu yapmak icin kullanilabilir.
pip3 list #Yuklu paketleri ve versiyonlarini gorebiliriz.
pip3 list --outdated #Guncel olanlari listelemek icin
pip install -r requirements.txt --upgrade #Bir txt dosyasindaki paket isimleri ile update yapabilirsin.

import pandas as pd

veri = pd.read_csv("https://bilkav.com/veriler.csv") #url ile de csv dosyalarini okuyabilirsin.
veri = pd.read_csv("./veriler.csv") #Hata alirsan calistigin dizinde ise dosya, dosya isminin basina ./ ekliyebilirsin.
veri = pd.read_csv("../veriler.csv") #Bir ust klasorde ise dosya isminin basina ../ eklemen lazim.
veri = pd.read_csv("~/Downloads/veriler.csv", encoding="utf-8") #encoding ile utf-8 karakter setini desteklemek icin kullanabiliriz.
veri = pd.read_csv("~/Downloads/veriler.csv", encoding="utf-8", engine="python") #engine python yapabiliriz.
#engine : "c", "python", "pyarrow"
veri = pd.read_csv("~/Downloads/veriler.csv", sep=",") #Farkli ayraclar kullanildi ise sep parametresine bunu belirtmelisin.
veri = pd.read_csv("~/Downloads/veriler.csv", index_col=0) #index_col baslik satirinin oldugunu belirmek icin ekliyoruz.
veri = pd.read_csv("~/Downloads/veriler.csv", header=None) #Veriyi header olarak alirsa header=None yapmaliyiz.
veri = pd.read_csv("~/Downloads/veriler.csv", na_filter=False) #Bos veriler yerine NaN yazmamasi icin kullaniliyor.
veri = pd.read_csv("veri.csv", skip_blank_lines=False) #Bos birakilan satirlarida almak icin skip_blank_lines=False yapiyoruz.
veri = pd.read_csv("veri.csv", skipfooter=2) #skipfooter ile sondan kac satir almak istemediginizi belirtiyorsunuz.
veri = pd.read_csv("veri.csv", skiprows=3) #skiprows ile bastan kac satiri almak istemediginizi belirtiyorsunuz.
#na_filter buyuk boyutlu dosylari okurken performan icin False verilebilir.Cunku NaN olarak doldurmadan bos birakiyor.
#Yukaridaki kisimlar hata alirsak kullanabiliriz.Hata almaz isek assagidaki kod ayni sekilde calisacaktir.
veri = pd.read_csv("~/Downloads/veriler.csv")
print(type(veri)) #<class 'pandas.core.frame.DataFrame'>
#Okumus oldugu csv dosyasini DataFrame olarak tutuyor.

url = "https://raw.githubusercontent.com/jokecamp/FootballData/master/Turkey/SuperLig/2015-2016/superlig.json"
veri = pd.read_json(url) #json file okuyabilirsin.Github'dan cekecegin datalari Raw basip linkini almalisin.
df = pd.DataFrame({"Name" : ["Aslan", "Kaplan", "Yilan"], "Yas" : [10, 11, 5]})
df.to_json("tojson.json", orient="records") #DataFrame json formatina donusturmek icin.
df.to_csv("tocsv.csv") #csv formatina cevirmek icin. DF to CSV

#DF to Excel
pip3 install openpyxl
import openpyxl
df.to_excel("out.xlsx")

#Open Document Format (ODF)
#Free/Ozgur Software dosya formatinda office dokumanlari kaydetmek icin.
pip3 install odfpy
import odf
veri.to_excel("out.ods") #ODF Spreadsheet (Calc) olarak excel kaydetmek icin.

print(veri.info()) #Daha fazla bilgi icin info() kullanabiliriz.
veri.info(verbose=True, show_counts=True) #Duzgun cikmadiginda bu sekilde calistirabilirsin.Cok col oldugunda gostermiyor tam.
print(veri.describe()) #R summary benzer sekilde istatistiksel olarak bilgiler gosterir.
print(veri.describe(include="all")) #Tum ozellikler icin include="all" parametresini vermemiz gerekiyor.
print(veri.describe(include=['object', 'float', 'int'])) #Veri tipleri ile hangi ozelliklerin dahil edilmesini belirleyebiliriz.
print(veri.describe(exclude=['object'])) #Benzer sekilde bu sefer exlude dahil etmemek icin kullanabiliriz.
print(veri.describe(percentiles = [0.10,0.50,0.80])) #Ceyreklikleri kendimiz verebiliriz. 
#percentiles parametresi 0 ile 1 arasinda degerler alir.
#info veri tipleri dogru algilayip algilamadigi kontrolu yapilabilir.
veri = pd.to_datetime(veri["tarih"]) #Datetime formatina cevirmek icin.
veri["cinsiyet"] = pd.to_numeric(veri["cinsiyet"]) #Bu islem True veya False sayisala cevirmiyor.
veri["cinsiyet"] = veri["cinsiyet"].astype(int) #True ve False degerleri 0 ve 1'lere cevirmek icin.
#Fakat hatali okunan sayisal degerleri pd.to_numeric ile cevirebilirsin.
print(veri.index) #Nasil indexlendigini bulmak icin index kullanabiliriz.
veri.set_index("yas") #Belirli bir col index olarak tanimlayabiliyorsun.
#Index'ler degistirlemezler tekara uzerinde oynamalar yapamazsiniz.

index1 = pd.Index([2,3,4,5,6,7,8,9])
index2 = pd.Index([0,1,2,3,4,5])
index1.intersection(index2) #index'lerin kesisimini bulamak icin.
index1.union(index2) #index'lerin birlesimini bulamak icin.
index1.difference(index2) #index'lerin farkini bulamak icin.
#Bunlari ogrenmek iki farkli veri setinde uyumsuzluklari veri kaybini veya cesitli sorunlarin yasanmasini onlemek icin.

veri["Adres"].str.split("/", expand=True) #/ ile ayrilmis bolumu parcalayip iki ayri stun yapmak icin.
veri["yeni"] = veri["cinsiyet"].astype(int)#Eger yeni die bir col yoksa ozelliklerin sonuna ayri stun olarak ekler.
#Araya stunu  eklemek icin insert kullanilir.
#.str.upper() bu yapiya benzer sekilde .str kullandiktan sonra string methodalarindan istedigimizi kullanabiliriz.
#expand = True oldugunda genisletip iki ayri col ayirir.
print(veri.memory_usage()) #Ne kadar hafiza kullandiklarina bakmak icin.
veri = pd.read_csv("~/Downloads/veriler.csv", usecols=[2,3]) #usecols ile istedigin col sadece alabilirsin.
veri = pd.read_csv("~/Downloads/veriler.csv", usecols=["boy","kilo", "cinsiyet"]) #col isimleri ile de usecols kullanabilirsin.
veri = veri.fillna(0) #Bos olan degerleri kendimiz deger atiyabiliriz.
veri.fillna(0, inplace= True) #Bu islem ile veri degerlerini degistiriyoruz.False kaldiginda copysinda degisikligi gosterir ana veriyi degistirmez.
veri = pd.read_csv("~/Downloads/veriler.csv") #.csv formatindaki dosyalari okumak icin.
veri.corr() #Verinin korelasyonuna bakmak icin
veri.corr(method="spearman") #Farkli methodlar ile korelasyonu hesaplamak icin.
"""
Korelasyon icin cesitli methodlar bulunmaktadir.
Method : "pearson", "kendall", "spearman"
"""

ser = pd.Series(list1) #Tek boyutlu listeyi pandas series cevirmek icin

ser = pd.Series(data= ["Ahmet",6, 3.3, "Test"], index= ["ad","sinif","not","bolum"])
#Sadece listeleri degil dict sozlukleri de yine pandas serisine donusturlebilir.
print(ser.values) #Value'lari sadece almak icin.
print(ser.keys()) #Key'leri sadece almak icin.
print(ser.index) #Yukaridaki islemle ayni sekilde key alabiliriz.
print(ser["ad"]) #Yine dict oldugu gibi degerlere erimek mumkun.
print(ser[["ad","not"]]) #Birden fazla degeri almak icin.
print(ser[[0, 2]]) #Yukaridaki islemlere benzer sekilde sayi vererekte ulasabiliriz.
#loc == location , iloc = index location
print(ser.loc['sinif']) #loc col ismine gore cagirabiliriz.
print(ser.iloc[1]) #iloc index degerine gore old. int deger girilmesi gerekiyor.
print(veri.iloc[1]) #Bu islem ile 1. indexteki satiri almak icin.
print(veri.iloc[:,1]) #Bu islem 1. stundaki degerlerin tamamini verir. : tamaminin alinacagini belirtir.
print(veri.iloc[:2,1]) # :2 oldugunda ilk 2 satiri almak istedigimizi belirtmek icin.
print(veri.iloc[4:,1]) # 4: ile 4. satirdan itibaren tamamini almak icin.
print(veri.iloc[:5,]) #head ile ayni sonucu verecektir.Ilk bes satir ve tum col alir.
print(veri.loc[:4,]) #loc yine benzer islemler ile head alabiliriz.
print(veri.iloc[-5:,]) #tail ile ayni sonucu verir.
print(veri.iloc[[2,4],]) # [] ile istedigimiz stun degerlerini belirtebiliriz.
print(veri.iloc[[2,4],[1,2,3]]) #Yine benzer sekilde col degerlerini de [] ile birden fazla col almak isteyebiliriz.
print(veri.iloc[2:5:2,1:4]) #Yukaridaki islemi aynisini yapar yani neyi almak istedigimizi istedigimiz gibi belirtebiliriz.
veri = veri.iloc[:, [1,2,3,4,5,6,7]] #Bu sekilde istediginiz kolonlari vererekte alabilirsiniz.
print(veri.loc[:4,["boy", "kilo"]]) #loc col isimleri ile alabiliriz.
print(veri.loc[veri.yas.notnull()]) #Ilgili col NaN veya null deger olmayanlari sadece gosterecek.
#notnull ilgili satirlari gostermez diger tum col null olsada gosterir.
ser["ad"] = "Mehmet" #Degeri degistirmek icin dict old. gibi
ser.iloc[1] = 4 #Yine loc veya iloc ile alip degeri degistirebiliriz.
data[np.logical_and(data["ilisim"] == secim, data["yil"] == secim2)]) #and ile birden fazla sorgu icin kullanabilirsin.
data[np.logical_or(data["ilisim"] == secim, data["yil"] == secim2)]) #or ile birden fazla sorgu icin birlestirebilirsin.
np.logical_not(3) #Olumsuzunu almak icin logical_not kullanabilirsin.
np.logical_xor(True, False) #xor almak icin logical_xor kullanabilirsin.
ser = ser.drop("bolum")
ser.drop("bolum", inplace=True) # inplace parametresini True yaparsak tekrar atama yapmamiza gerek kalmadan siler.
veri.drop(["_id", "DATE_TIME"], axis= 1, inplace=True) #Belirlemis oldugum col silmek icin.
veri.drop(columns=["yas","cinsiyet"], inplace=True) #Yukaridak isilemin benzerini axis vermeden columns ile yapabilirisin.
#inplace False oldugunda ayri bir copy'sinda bu islemi yapar degisikligi daha sonra seriye yansitmaz.
print(ser.sort_index()) #Index yani key degerlerine gore siralar.Fakat atama yapmazsaniz seri ayni kalir.
print(ser.sort_values()) #Degerleri siralamak icin.Fakat atama yapmazsaniz seri ayni kalir.
#Not: int ve  string degerler var ise hata verir.
df.to_string() #DataFrame string cebvirmek icin.

parcala = veri["DATE_TIME"].str.split(" ", expand=True)
parcala["DATE"] = parcala[0]
parcala["TIME"] = parcala[1]
parcala.drop([0,1], axis = 1, inplace=True)
parcala["DATE"] = pd.to_datetime(parcala["DATE"])
#parcala["TIME"] = time.isoformat(parcala["TIME"])
veri.insert(1, column="DATE",value= parcala["DATE"])
veri.insert(2, column="TIME",value= parcala["TIME"])
veri.drop(["_id", "DATE_TIME"], axis= 1, inplace=True)

print(type(ser)) #pandas.core.series.Series
print(ser) #Seriyi index vererek ekrana yazdiriyor.Veri ayni kaliyor yalniz.
#Dictlerde ise key ve value'lara gore ekrana cikti verir.
#Not pandas numpy farkli olarak fakli veri yapilarini destekler tip donusumu yapmadan orjinal veri tipi halini korur.
#Fakat dtype: object olarak tutuyor genel olarak farkli veriler oldugundan dolayi
#Tek bir veri tipinde oldugunda genel olarak ornek dtype: int64 olarak tanimliyor.
print(ser.shape) #Sekilini gormek icin.
print(ser.ndim) #Kac boyutlu oldugunu gormek icin.
print(ser.size) #Kac eleman oldugunu gormek icin.
print(ser.name) #Ismini gormek icin.Eger bir atama yapilmadiysa None doner.
ser.name = "Test1" #Series isim atamasi yapmak icin.
print(ser.dtype) #Veri tipini gormek icin.
print(sorted(ser)) #Siralamak icin sorted() kullanilabilir fakat atama yapilmazsa seri ayni kalir.
print(sorted(ser, reverse=True)) #Tersten siralamak icin.
#Not: int ve  string degerler var ise hata verir.
print(ser.sum()) #Serinin elemanlarinin toplami icin.
print(ser.mean()) #Ortalama almak icin.
print(ser.product()) #Carpimlarini bulmak icin.
print(ser.max()) #En buyuk degeri bulmak icin.
print(ser.min()) #En kucuk degeri bulmak icin.
print(veri.columns) #Col isimlerini listeler.
print(veri.columns[1]) #Hangi indexteki col isimini almak icin kullanilir.
print(veri.head()) #Ilk 5 satiri almak icin.
print(veri.tail()) #Son 5 satiri almak icin.
print(veri.head(10)) #Deger girdigin o kadar kismi aliyor.10 old. ve head old. ilk 10 satiri alcak.
print(veri.sort_values("boy").head()) #Bu islem ile degeleri siraladik boy col gore ilk 5 tanesini aldik.
veri.sort_values("boy", inplace=True) #Verinin siralamasini degistirmek istersen col degerine ek inplace = True vermelisin.
veri.sort_values("boy", ascending=False, inplace=True) #Buyukten kucuge yapmak istersne ascending = False yapmalisin.
print(veri.get("cinsiyet")) #col name gore degerleri alabiliriz.
print(veri.get("is")) #Eger olmayan bir col veya key verdiginde ise None doner.
print(veri.get("is", default="Bulamadim.")) #default degerini degistirerek None yerine bir mesaj verebiliriz.
print(veri.get("is", default=-1)) #Deger de gosterebiliriz.
bilgi = pd.DataFrame(bilgi) #DataFrame cevirmek icin.
#Dict index degerlerini vermez isek hata verir DataFrame donusturemez.
#Birden fazla dict degeri ekledigimizde index farkli oldugunda olmayan indexlere NaN degeri atar.

print(veri["boy"].head())
print(veri["boy"][1]) #Bircok farkli sekillerde istedigim degeri alabiliriz.
print(type(veri["boy"].head())) #<class 'pandas.core.series.Series'>
#Ikisinin farki DataFrame olan iki boyutlu ve col name ile gosterirken.Series'de ise bize sadece satiri tek boyutlu getirir.
print(veri[["boy"]].head())
print(veri.boy) #Col isimileri ile de degerleri alabilirsin.
print(veri.cinsiyet.astype("int64")) #Col degerlerin veri tipini degistirebiliriz.
print(type(veri[["boy"]].head())) #<class 'pandas.core.frame.DataFrame'>
veri["renk"] = ["kumral", "esmer","beyaz"] #Yeni bir stun eklemek  icin.Fakat liste satir sayisina esit olmamalidir.
veri["vki"] = veri["kilo"] / (veri["boy"] / 100 ** 2) #Stunlari hesaplayip ekleme yapabiliriz.
veri.pop("vki") #pop ile stunu silmek icin.pop listedeki kullanimdan farkli olarak col ismini vermemiz gerekiyor.
veri = veri.drop("vki", axis=1) #Yukaridaki ayni islemi drop ile de yapabiliriz.
veri = veri.drop([0,2,3], axis=0) #axis =  0 vererek satirlari indextekileri gore silebiliriz.
#Satirda oldugu gibi stunlarida [] icine alarak birden fazla col silebiliriz.
print(veri.sum()) #Toplamlari bulmak icin col bazli olarak toplar.
print(veri.isnull()) #Veride null olan degerleri True digerlerini False yapip tum veriyi gosteriyor.
print(veri.isnull().sum()) #Kac tane null deger var onlari gormek icin.col gore ayri ayri hesapliyor.
print(sum(veri.isnull().sum())) #Tum veride kac tane null deger oldugunu bulmak icin.
print(veri.isnull().sum().sum()) #Yukaridaki islemin yanisini yapar.
print(veri.isna().sum()) #NA degerleri kac tane oldugunu bulmak icin.
print(sum(veri.isna().sum()))
print(veri.isna().sum().sum())
print(veri.notnull()) #notnull isnull tam tersidir.
print(veri.notnull().sum())
print(sum(veri.notnull().sum()))
print(veri.notnull().sum().sum())
print(veri.notna())
print(veri.notna().sum())
print(sum(veri.notna().sum()))
print(veri.notna().sum().sum())
veri = veri.dropna() #dropna() ile NA olan degerleri verimizden silmek icin kullaniyoruz.
veri.dropna(inplace=True) #inplace atama islemi olmadan parametre olarak belirtmek icin silinip veride de degistirmesi icin.
#dropna() verideki NaN olan, girilmeyen, null degerleri silmek icin kullaniliyor.
veri.dropna(how="all",inplace=True) #how ile nasil NaN oldugunda silmesi icin.all ile ilgili satirin tum col degerleri bos olmali.
veri.dropna(how="any",inplace=True) #any bir tane bile bulsa siler.
veri.dropna(how="any", axis=1 ,inplace=True) #axis = 1 stundaki bos olan kisimlara bakiyor. 
#how any oldugundan bir tane bu yuzden tum col silecek.
veri.dropna(how="all", axis=1 ,inplace=True) #Tum col bos ise bunu silecek.
veri.dropna(how="all", axis=0 ,inplace=True) #Tum ilgili satir bos ise bunu silecek.
#dropna Default degerleri : how = any , axis = 0
veri.reset_index(drop=True, inplace=True) #index'i silmek icin. 
veri = veri.dropna(thresh=3) #thresh parametresi ile belli bir NaN sayida olmadikca o satiri silmiyor bunun icin kullaniliyor. 
veri = veri.drop_duplicates() #drop_duplicates() ile tekrar eden verileri silmek icin kullaniyoruz.
veri.drop_duplicates(subset="yas",inplace=True) #Belli bir col tekrarlilari silme icin subset = ilgili col atanir.
print(veri["yas"].unique()) #Essiz olarak degerleri gormek icin unique kullaniyoruz.Bir kere gosteriyor.
print(veri["yas"].nunique()) #Kac tane tekil deger var onu goremek icin.
veri = veri.fillna(1) #Bos olan degerleri istedigimiz bir degerle doldurmak icin.
veri = veri.fillna(method="ffill",axis=0) #ffill bir onceki degeri alir.axis = 0 old. col bir ust satirin degerini alir.
#ffill axis = 1 yapmak dogru olmayabilir.Cunku alakasiz bir deger gelebilir.
veri = veri.fillna(method="ffill") #Bir onceki degeri almak icin direkt bunu da yazabilirsin.
veri = veri.fillna(method="backfill") #Bir sonraki degeri almak icin.Son satirda NaN varsa o oyle kaliyor.
veri = veri.fillna(method="bfill") #Bir sonrakini yine aldi ama farki nedir bilmiyorum.
veri["cinsiyet"] = veri["cinsiyet"] + 1 #Matematiksel islemleri yine yapabiliyoruz.
print(veri["cinsiyet"].value_counts()) #Degerlerden kac tane oldugunu gormek icin.
print(veri["cinsiyet"] == 1) #1 esit olanlari True olmayanlari False olarak tek satirda gosterecek.
print(veri[veri["cinsiyet"] == 1]) #Sadece cinsiyeti 1 olanlari gormek icin.

maskorfilter = veri["cinsiyet"] == 1
print(veri[maskorfilter]) #Ayni ciktiyi veriyor fakat filtreleme islemini ayrci tutarak baska sorgular icin de kullanabiliyoruz.
print(veri2[maskorfilter]) #Baska bir DataFrame icin de yine calisacaktir.Yani bir kere filtre olusturup kullanabiliriz.
#Not: Yukaridaki ozelligi numpy'da destekler.

print(veri.isin([1])) #1 icerip icermedigini kontrolu icin.DataFrame True veya False olarak doldurur.
print(veri[veri["boy"].between(160,190)]) #between ile belli bir araligi alabilir.
print(veri[~veri["yas"].duplicated()]) #duplicated tekrar olup olmadigini kontrol eder. ~ False degerleri almak icin.
#~ tekrari olsada ilkini alir.Cunku duplicated False degeri atar eger onceki degerden bir daha buldu ise onu True degeri atar.
veri = pd.DataFrame() #Bos bir DataFrame olsuturmak icin.
seri.to_frame() #to_frame ile DataFrame cevirebiliriz.Pandas serilerini dahil.
veri.groupby("GEOHASH").mean() #grouoby gruplamamiza yariyor. Grouplamak istedigimiz ozelligi belirtiyoruz.
#mean ortalamasini alarak grupluyoruz.Baska sekilde de gruplayabiliriz.
pd.get_dummies(veri, columns= ["cinsiyet"], drop_first=True,dummy_na=True,) #Kategorik verileri sayisal verilere cevirmek icin.
#drop_first ile ilk olan col siliyor bunu yapmazsak iki tane col ayni ozellik icin oluyor.
pd.get_dummies(veri, columns= ["cinsiyet"], drop_first=True,dummy_na=False,) #na stunu olsun mu olmasin mi yi dummy_na yapiyoruz.
#Dummy Variable == Kukla Degisken ayni seyi ifade eden degiskenlerdir.Bu col tekrar ettigi icin makine ogrenmesi modellerini yaniltabilir.
#Bazi makine ogrenmesi modelleri bu duruma kendini adapte edebilmis olmaisina ragmen dikkatli olunmali modeli yanlis egitilmemeli.
print(pd.date_range("12-12-2012", "10-11-2023")) #Belirli bir tarih araligi icin data_range kullanip islem yapabiliriz.

veri["DATE_TIME"] = veri["DATE_TIME"].str.split(" ") #Parcalamak gerekebiliriz oncesinde
print(veri.explode("DATE_TIME")) #Parcaladigimiz kisimlari ayri satirlar olarak gormek icin explode kullaniyoruz.

veri.columns = ["boylar", "kilolar", "yaslar", "cinsiyetler"] #Kolon isimlerini degistirmek icin.
veri.rename(columns={"boy": "boylar", "kilo" :"kilolar"}, inplace=True) #rename ile yine col isimlerini degistirebiliriz.
#Ikisinin farki rename istedigimiz col sadece yazarken veri.columns ise tum col isimlerini degistirmesekte eklemek zorundayiz.

cevir = lambda x : x / 100
veri["boy"] = veri["boy"].apply(cevir) #Yazmis oldugumuz fonksiyonlari apply ile uygulayabiliriz.
veri["boy"] = veri["boy"].map(lambda x : x / 100) #map ile uygulayabiliriz.
print(veri.applymap(arttir)) #Tum veri setine islemi uygulamak icin applymap() kullaniyoruz.

print(veri.groupby("cinsiyet").agg(["mean", "median", "std"])) #agg ile uygulamak istedigimiz yontemleri belirtebiliriz.
print(veri.groupby("cinsiyet").agg(["sum", "max", "min"])) #Bircok methodu uygulayabiliriz.
veri.groupby("cinsiyet").sum() #Tum ozelliklerde cinsiyetlere gore toplamlarini verecek.
veri.groupby("cinsiyet").mean()["boy"] #Bu islem ile sadece boy ozelligindeki islemin sonucunun ne olacagini bakabiliriz.
veri.pivot_table("boy", index="cinsiyet", aggfunc="mean") #Yukaridaki islemin aynisini tablo seklinde yapmak icin.
veri.pivot_table("boy", index="cinsiyet", columns="yas",aggfunc="count")#columns ile stunlarda hangi ozelligin olacagini belirtiyoruz.
#columns ile farkli ozellikleri birlikte isleme tabi tutarak bakabiliriz.
#Yukaridaki islem benzer bircok islem veriyi anlamak icin guzel bir tablo olusturmaya yarayabilir.
veri3 = pd.concat([veri, veri2]) #Verilerimizi birlestirmek icin concat kullanabiliriz.
veri3 = pd.concat([veri, veri2], axis=1) #axis parametresini 1 verirsek col bazli olarak birlestirir.
npb = veri["boy"].to_numpy() #numpy cevirmek icin to_numpy kullaniyoruz.
veri.to_csv("dosya_adi.csv") #csv dosyasi olarak kaydetmek icin to_csv kullaniyoruz.
veri.to_csv("~/Downloads/data/veri.csv") #Farkli bir dosya kaydetmek icin path verilmelidir.
clip = pd.read_clipboard() #Son kopyalanmis veriyi okuyabilirsin.
veri.to_clipboard() #Veriyi clipboard aktarmak icin to_clipboard() kullanilir.

#Gercek ve tahmin degerlerini gormek icin.
df = pd.DataFrame({"Gercek":y_test.flatten(),  "Tahmin": y_pred.flatten()})
print(df)

print(df.tail(-1)) #Tum verileri gosterir fakat tail old sondan gosterir.
df.sample() #Bir ornek almak icin
df.sample(10)  #Kac tane ornek almak istersek o kadar degeri giriyoruz.

veri.plot()
veri.plot(kind= "hist")
"""
The kind of plot to produce:
        "line" : line plot (default)
        "bar" : vertical bar plot
        "barh" : horizontal bar plot
        "hist" : histogram
        "box" : boxplot
        "kde" : Kernel Density Estimation plot
        "density" : same as ‘kde’
        "area" : area plot
        "pie" : pie plot
        "scatter" : scatter plot (DataFrame only)
        "hexbin" : hexbin plot (DataFrame only)
"""

#PandasAI
pip install pandasai

from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
llm = OpenAI(api_token="YOUR_API_TOKEN") #OpenAI aldigin token girip sorgu yapabilirsin.

pandas_ai = PandasAI(llm, conversational=False)
given_prompt = "Which one most sale property"
pandas_ai(df, prompt=given_prompt) #df ile veri veriyoruz.prompt ile yapcagimiz sorguyu veriyoruz.
Prompt text input ile kullanicidan alarak chatgpt kendi sitene entegre edebilirsin.
PandasAI(llm, save_charts=True) #Grafigi kaydetmek icin.Grafikler ./pandasai/exports/charts yoluna kaydediliyor.
PandasAI(llm, enforce_privacy = True) #enforce_privacy = True parametresi ile gizlilik saglanabilir.
pandas_ai([employees_df, salaries_df], prompt=given_prompt) #Birden fazla veri seti ile islem yapmak icin.

#Pandas Profiling
pip install pandas-profiling
pip install ipywidgets #HTML dosyalari notebookta gormek icin
import ipywidgets
import pandas_profiling as pp # Kesifsel Veri Analizi (EDA) icin bir kutuphanedir.
pp.ProfileReport(veri) #Veri setinin tum raporunu gormek tek tek veri anlama kodlarina gerek kalmadan korelasyonu dahil gosteriyor.
report = pp.ProfileReport(veri)
report.to_file("rapor.html")

"""
#Terminal ayarlari
pai --help
pai [OPTIONS]

Options:

    -d, --dataset: The file path to the dataset.
    -t, --token: Your HuggingFace or OpenAI API token, if no token provided pai will pull from the .env file.
    -m, --model: Choice of LLM, either openai, open-assistant, starcoder, or Google palm.
    -p, --prompt: Prompt that PandasAI will run.


#Terminalde calistirmak icin bir ornek
pai -d "~/pandasai/example/data/Loan payments data.csv" -m "openai" -p "How many loans are from men and have been paid off?"
"""

.env #Bu dosyayi olusturup bazi ayarlari aktarabilirsin bu gitignore icin faydali olabilir.
# OpenAI
llm = OpenAI(api_token="YOUR_API_KEY")

# Starcoder
llm = Starcoder(api_token="YOUR_HF_API_KEY")



#Matplotlib
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
%matplotlib notebook #Bu satir ile interaktif olarak gosterecek grafikleri
%matplotlib inline #Bu satir ile statik gosterimde bulunacak.Daha farkli gosterimler mevcut.
%lsmagic #Daha fazla sihirli fonksiyonlar icin bu komut ile bakabilirsin. 
plt.plot([2,4,6,8,10], [1,3,5,7,9]) #Kendimiz x ve y degerleri vererek bir plot cizdirtebiliriz.

x = [2,4,6,8,10]
plt.plot(x, np.cos(x), label= "cos") #Etiketlemek ve lejantta gormek icin label kullaniyoruz.
plt.plot(x, np.sin(x), label= "sin")
#Birden fazla grafigi ayni anda cizdirmek mumkun.Sadece plot degil diger grafiklerde de destekliyor.
plt.legend() #lejant gorstmek icin.
"""
legend loc parameter values:
Location String             Location Code
'best' (Axes only)          0
'upper right'               1
'upper left'                2
'lower left'                3
'lower right'               4
'right'                     5
'center left'               6
'center right'              7
'lower center'              8
'upper center'              9
'center'                    10

fig = plt.figure()
fig.legend() loc paramtre values:
'outside upper left',
'outside upper center',
'outside upper right',
'outside lower left',
'outside lower center',
'outside lower right'
"""

plt.plot(x, np.cos(x), label= "cos", linestyle= "--") #linestyle ile cizgilerin nasil gosterilmesini belirtebiliyoruz.
plt.plot(x, np.sin(x), label= "sin", linestyle= "-.")
"""
linestyle:
'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
destekledigi cizimlerdir.
"""
plt.plot(x, np.cos(x), label= "cos", color = "blue") #color parametresi ile renkleri kendimiz elle atayabiliriz.
plt.plot(x, np.cos(x), label= "cos", color = "blue", linewidth = 5) #linewidth ile cizginin kalinligini degistirebiliriz.
plt.plot(x, np.sin(x), label= "sin", color = "black", alpha = 0.5) #alpha parametresi ile saydamligini degistirebiliriz.
plt.plot(x, np.cos(x), label= "cos", marker = "o") #Marker ile degerleri isaretleyebiliriz.
"""
All possible markers are defined here:
"."                     point
","                     pixel
"o"                     circle
"v"                     triangle_down
"^"                     triangle_up
"<"                     triangle_left
">"                     triangle_right
"1"                     tri_down
"2"                     tri_up
"3"                     tri_left
"4"                     tri_right
"8"                     octagon
"s"                     square
"p"                     pentagon
"P"                     plus (filled)
"*"                     star
"h"                     hexagon1
"H"                     hexagon2
"+"                     plus
"x"                     x
"X"                     x (filled)
"D"                     diamond
"d"                     thin_diamond
"|"                     vline
"_"                     hline
0                       (TICKLEFT)
1                       (TICKRIGHT)
2                       (TICKUP)
3                       (TICKDOWN)
4                       (CARETLEFT)
5                       (CARETRIGHT)
6                       (CARETUP)
7                       (CARETDOWN)
8                       (CARETLEFTBASE)
9                       (CARETRIGHTBASE)
10                      (CARETUPBASE)
11                      (CARETDOWNBASE)
"none" or "None"        nothing
" " or ""               nothing
'$...$'                 Render the string using mathtext. E.g "$f$" for marker showing the letter f.
"""
plt.style.use("ggplot") #Stil vererek gorseli tamamen degistirebiliriz.
#Not: yazilan kodlarin sirasi onemli yoksa grafigi gosterirken girilmis oldugumuz kodlarin bazilarini uygulamadan gosterecektir.
"""
plt.style.available 
print(plt.style.available) 
#Yukaridak iki kod ile style eklebilecegin degerleri gorebilirsin.
['Solarize_Light2',
 '_classic_test_patch',
 '_mpl-gallery',
 '_mpl-gallery-nogrid',
 'bmh',
 'classic',
 'dark_background',
 'fast',
 'fivethirtyeight',
 'ggplot',
 'grayscale',
 'seaborn-v0_8',
 'seaborn-v0_8-bright',
 'seaborn-v0_8-colorblind',
 'seaborn-v0_8-dark',
 'seaborn-v0_8-dark-palette',
 'seaborn-v0_8-darkgrid',
 'seaborn-v0_8-deep',
 'seaborn-v0_8-muted',
 'seaborn-v0_8-notebook',
 'seaborn-v0_8-paper',
 'seaborn-v0_8-pastel',
 'seaborn-v0_8-poster',
 'seaborn-v0_8-talk',
 'seaborn-v0_8-ticks',
 'seaborn-v0_8-white',
 'seaborn-v0_8-whitegrid',
 'tableau-colorblind10']
"""
plt.xlim(0,5) #x kordinatinda limitlemek icin.
plt.ylim(0,5) #y kordinatinda limitlemek icin.
#Bu fonksiyonlar grafigin belli bir kismini gosterir bu islem kirpma gibi de dusunebilirsiniz.Bazen grafik dogru gozukmeyebiliriz.
plt.axis("tight") #Otomatik limit belirleyip sinirlandirmak istersek.
plt.xlabel("apsis") #x kordinatina isim vermek icin.
plt.ylabel("ordinat") #y kordinatina isim vermek icin.
plt.title("Degerler Grafigi") #Grafigimize bir baslik vermeki icin.

fig = plt.figure()
plt.plot(x, np.cos(x), label= "cos", color = "blue", marker = "o")
fig.savefig("grafik.jpg") #savefig ile bir resim dosyasi olarak kaydetmek icin.Dosya ismini vererek kaydetebiliriz.
fig.savefig("grafik.jpg", transparent=True) #transparent parametresi ile arkaplandan kurtulmak icin kullanabilirsin.
fig.savefig("grafik2.jpg", bbox_inches = "tight") #bbox_inches = "tight" parametresi ile kenar boluklari olmadan kaydetebiliriz.
fig = plt.figure(figsize=[3,3], dpi=300) #figsize ile boyutunu, dpi parametresi ile kac dpi olacagini belirtebiliyoruz.
#Bu degeleri verdigimide grafik biraz farkli cikiyor.Sol azcik kirpilmis cikti ornek olarak.Dogru degerler verilmeli.
plt.bar(x, x, color= "blue") #Bar ile cubuk grafigi ekliyebiliriz.
plt.bar(x, x, color= "blue", width=0.5) #width ile cubuklarin genisligini verebiliriz.
plt.bar(x, x, color= "blue", align='edge') #align hizalamak icin. edge koseye gore center merkeze gore.Default center.
plt.barh(x, y) #barh dikey degil yatay olarak cubuk grafikleri gostermek icin.
plt.scatter(x,y) #scatter plot cizdirmek icin scatter kullaniyoruz.
#align farkini anlamak icin gridlere bakmalisin.
plt.xticks(x, ["kucuk","orta", "kobi", "buyuk", "holding"]) #xticks cubuklarin her birine isim vermek icin kullaniyoruz.
plt.yticks(y, ["esnaf", "kobi", "sirket", "fabrika", "holding"]) #yticks ile y degerlerine isim vererbilirsin.
#Not kac tane cubuk var ise o kadar isim olmali yoksa hata veriyor.
plt.xscale("log") #x degerlerini olceklendirmek icin cesitli methodlar ile yapabiliriz.
plt.yscale("log") #y degelerini de yine ayni sekilde
"""
scale parameter values:
'linear', 'log', 'symlog', 'asinh', 'logit', 'function', 'functionlog'
"""
plt.hist(x, density=True) #Density ile olasiligi veya orani gostermek icin kullaniyoruz.%'lik olarak gosteriyor.
plt.hist(x, density=True, bins=100) #bins ile dagilimi genisletmek diger degerleri de gormek icin arttirabilirsin.
#Deger araligini kucultup daha cok degeri gostermemize olanak taniyor.
plt.hist(x, density=True, cumulative=True) #Kumulatif toplamlari gormek icin cumulative=True yapiyoruz.
plt.hist(x, align='mid') #align hizalamak icin kullaniyoruz.
#align: 'left', 'mid', 'right'
plt.hist(x,orientation='horizontal') #Grafik yatayda gostermek icin or orientation="horizontal" yaptik.Default "vertical"
plt.hist(x,histtype="step") #histogramin tipini degistirmek icin.
#histtype: 'bar', 'barstacked', 'step', 'stepfilled'
plt.boxplot(x) #Kutu grafigini cizdirmek icin.
plt.boxplot(x, notch= True) #notch R'daki gibi Q2 dogru ice bukuyor kutuyu

plt.subplot(1, 2, 1) #subplot ile ayni anda birden fazla grafigi yan yana veya alt alta gosterebiliyoruz.
plt.hist(x)
plt.subplot(1, 2, 2) #subplot (satir , stun , konum) 
plt.boxplot(x, notch= True)
plt.subplot(2, 2, 2).text(1.5,1.5,"yazi") #Grafik uzerine yazi eklemek icin plt.text'de kullanabilirsin.
plt.text(5,5,"yazi", bbox= {facecolor: "green", alpha: 0.4}) #alpha ile opakligini facecolor ile rengini verebiliriz.

g3 = plt.subplot2grid((2,2),(1,0),rowspan = 2, colspan = 2) #subplot2grid ile grafigi rowspan ve colspan parametreleri ile genisletebiliriz.

plt.stem(x, y) #Scatter plot cubuklu halini ciziyor.
plt.step(x, y) #Adindan da anlasilaricagi gibi adimlar seklinde gosteriyor.
plt.fill_between(x, y, y2) #SVM benzer bir araligi gorebilecek sekilde doldurmaya yariyor.
plt.hist2d(x,y) #Histogrami farkli gostermek icin.
plt.pie(veri["cinsiyet"]) #Pasta grafigi icin pie kullaniyoruz.
plt.hexbin(x, y) #x, y, C die bir parametre daha eklenmis ornekte fakat nedenini bilmiyorum.
plt.errorbar(x,y) #Ornekte xerr, yerr parametreleri verilmis yine tam bilmiyorum.
plt.violinplot(veri["yas"]) #Verilerin nasil dagildigini farkli bir bicimde gosteriyor.
plt.eventplot(veri["yas"]) #Cubuk grafigine benzer bir grafik gosteriyor.
#Yukarida ele aldikarimiz istatistiksel grafikler bazilari verileri belli araklarida tutmus bunlarin nedeni ogrenmeli.
#Yukaridaki grafikler ne icin kullaniliyor cikti gorduklerimiz ne anlama geliyor ogrenmek gerekli.

#Seaborn
import seaborn as sns
sns.set() #Bu komutu matplotlib cizdirmek istedigimiz komutlarin basina eklersek seaborn ile cizdirecektir.Bu daha iyi sonuc icin
veri = sns.load_dataset("penguins") #Seaborn hazir veri setini yuklemek icin.
"""
load_dataset all dataset names:
['anagrams',
 'anscombe',
 'attention',
 'brain_networks',
 'car_crashes',
 'diamonds',
 'dots',
 'dowjones',
 'exercise',
 'flights',
 'fmri',
 'geyser',
 'glue',
 'healthexp',
 'iris',
 'mpg',
 'penguins',
 'planets',
 'seaice',
 'taxis',
 'tips',
 'titanic']
"""

sns.set_theme()
sns.set(rc= {"figure.dpi" : 300, "figure.figsize" : (5,5)}) #Gragimizin dpi ve boyutunu ayarliyabiliriz.
#300 dpi uluslararasi genel bir standart degerdir.Tatmin edilebilir olcude grafikler gosterip kaydedebilirsiniz.
sns.scatterplot(x="boy", y="kilo", data= veri, hue= "cinsiyet") #hue ile renklendirmeyi hangi ozellige gore olacagini belirtiyoruz
sns.histplot(x="boy", data= veri)
sns.histplot(x="boy", data= veri, binwidth=2) #binwidth ile cubuklarin genisligini belirtmek icin kullaniyoruz.
sns.histplot(x="boy", data= veri, kde= True) #kde parametresi ile olasiligik dagilim egrini cizdirmek icin True yapiyoruz.
sns.histplot(x="boy", data= veri, hue= "cinsiyet") #Yine renklendirmek icin hue kullaniyoruz.Bu cubuklari ayni zamanda parcaliyor.
sns.histplot(y="boy", data= veri) #y esitlersek dikey degil yatayda grafigi cizer.
sns.barplot(x="boy", y="kilo", data= veri) #Barplot icin kendisi diger barplotlardan farkli olarak renklendiriyor.
#Bu renklendirme yerine hue paranetresi de eklenebilir.
sns.boxplot(x="cinsiyet", y="kilo", data= veri) #Kutu grafigi cizdirmek icin.
sns.violinplot(x="cinsiyet", y="kilo", data= veri) #Kutu grafigine benzer sekli farkli bir grafik cizdirmek icin.
sns.distplot(veri.boy) #Histogram grafiginin density ve kde olarak ozellestirilimis halidir.
sns.FacetGrid(veri, col= "boy", row="kilo").map(sns.distplot, "yas") #FacetGrid ile subplot gibi calisir.
#Fakat yukaridaki kod calistiramadi daha uygun col ve row bulmali o sekilde cizdirilmelidir.
sns.pairplot(veri, hue = "cinsiyet") #Veriler arasindaki ikili iliskiyi gormek icin birden fazla grafik olusturur.
sns.pairplot(veri, hue = "cinsiyet", height=4) #height ile grafigi boyutlandirabilirsin.size parametreside var.
sns.pairplot(veri, hue = "cinsiyet", hue_order=["e","k"]) #hue order ile sira verebilirsin.Bu sira renk icin de degisiklik oluyor.
sns.pairplot(veri, hue = "cinsiyet", kind="hist") #Kind ile tum grafiklerin nasil cizilecegini
sns.pairplot(veri, hue = "cinsiyet", diag_kind="hist") #diag_kind ile grafik gosterimlerinin nasil olacagini belirtir.
sns.heatmap(veri.corr()) #Isi grafigi icin heatmap kullaniyoruz.Ozelliklerin korelasyonlarina bakabiliriz.
#Not: Korelasyon icin tum ozellikler sayisal degerde olamalilar.
sns.heatmap(veri.corr(), annot=True) #annot parametresini True yaparsak degerleri uzerinde goruruz.
sns.kdeplot(veri.boy, shade= True ,color= "blue") #kde grafigi cizdirmek icin kdeplot kullaniyoruz.
#kdeplot shade parametresi golge olup olmayacagini , color parametresi renginin ne olacagini belirmek icin.
sns.jointplot(x="boy",y= "kilo",data=veri, kind= "reg") #Joinplot ile birlesik grafikler gosterebiliriz.
#catplot grafigi de var fakat tam olarak ne icin kullanildigini bilmiyorum.

#SciPy
from scipy.fft import fft, fftfreq #fft == Fast Fourier Transforms icin
from scipy import stats #Istatistiksel hesaplamalar icin.
z = np.abs(stats.zscore(veri.kilo)) #zscore hesaplamak icin stats.zscore kullaniyoruz.
#Z-Score'lara bakarak aykiri degerleri bulup veri setinden cikarabilirsin.

#Artificial intelligence (AI) == Yapay Zeka
#Lazy Learning Big Data icin daha sik kullanilan bir ogrenme yontemidir.
#Eager learning kullanildigi ve lazy learning gore avantajlari ve dezavantajlari mevcuttur.
"""
                        Veriler
        Kategorik                       Sayisal
    Nominal Ordinal             Oransal(Ratio) Aralik(Interval)

Verilerin farkli olmasi kullanacagimiz makine ogrenmesi yontemini de belirliyor.
Kategorik veriler icin siniflandirma, sayisal veriler icin regresyon veya tahmin yontemlerini kullaniyoruz.
"""
#Yapay Zeka en cok kullanilan yontem Cross-industry standard process for data mining (CRISP-DM)'dir.
#Tahmin gecmis ve gelecek icin ongoru sadece gelecek icin.

#Machine learning (ML) == Makine Ogrenmesi
from sklearn.impute import SimpleImputer #Eksik verileri duzenlemek icin kullanacagimiz method icin
veri = pd.read_csv("https://bilkav.com/eksikveriler.csv")
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean') #eksik degerlerin ne oldugunu belirtmek icin np.nan kullandik.
#Eksik degerler yerine farkli degerlerde yazilabilir.Ornegin negatif deger sizin icin eksik bir veri olabilir.
#Strategy ise ortalamasinin alinmasi icin mean olarak verdik.
"""
The imputation strategy:
"mean",
"median",
"most_frequent",
"constant" ([then replace missing values with fill_value. Can be used with strings or numeric data.])
"""

cols = veri.iloc[:,1:4].values #Degisiklik yapmak istedigimiz stunlari belirtiyoruz.
imp_mean = imp_mean.fit(cols[:,1:4]) #fit ile impute egitiyoruz.
cols[:,1:4] = imp_mean.transform(cols[:,1:4]) #Ortalama ile degistirmek icin transform ile degisiklik yaptiriyoruz.
print(cols)
imp_mean = SimpleImputer(missing_values=np.nan, fill_value= 20,strategy="constant") #fill_value degeri vermez isen 0 atiyor.

from sklearn import preprocessing
ulkeler = veri.iloc[:,0:1]
le = preprocessing.LabelEncoder()
ulkeler.iloc[:,0] = le.fit_transform(veri.iloc[:,0:1]) #Bu islem sayesinde kategorik olan ulke isimlerini sayisal cevirdik.
print(ulkeler)

veri = veri.apply(preprocessing.LabelEncoder().fit_transform) #LabelEncoder tek satirda tum veriye uygulayabilirsin.
#Dikkat OneHotEncoder() yapmak istedigin kolonlarda LabelEncoder yapilmis olabilir.

ohe = preprocessing.OneHotEncoder()
ulkeler = ohe.fit_transform(ulkeler).toarray() #Bu islemde ulkeleri one hot encoder methodu ile daha dogru bir cevirime yariyor.
print(ulkeler)
#OneHotEncoder kolon basliklarini header tasiyip 1 0 var yada yok olarak kategorik veriyi sayisal cevirmek icin kullanilan yontem.

df1 = pd.DataFrame(data=ulkeler, index = range(22), columns = ['fr','tr','us'])
df2 = pd.DataFrame(data=cols, index = range(22), columns = ['boy','kilo','yas'])
cinsiyetler = veri.iloc[:,-1].values
df3 = pd.DataFrame(data = cinsiyetler, index = range(22), columns = ['cinsiyet'])
yv =pd.concat([df1,df2], axis=1)
yv2=pd.concat([yv,df3], axis=1)
veri = yv2

#Principal Component Analysis (PCA) boyut indirgeme , donusturme , veya oznitelikleri birlestirme islemleri icin kullanilir.
#Egien Value == Oz deger, Egien Vector == Oz yoney   Av= Xv bu islem bir vektro skaler ile caripimina indirgenebildigini belirtir
from sklearn.decomposition import PCA
pca = PCA(n_components=2) #n_components ile kac boyuta indrigemek istedigimizi belirtiyoruz.
#Verdigimiz boyut sayisi ile iki tane oznitelige indirdi fakat bu kayipa da neden olabilir bu yuzden bu sayi arttirilabilir.
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test) #Test icinde donusum yapilmali fakat X_train egitilmise gore yoksa boyutlar birbiri ile uyusmayacaktir.

#Linear Discriminant Analysis (LDA) , PCA ile benzer sekilde boyut indirgeme icin kullanilan baska bir yontemdir.
#PCA'den farki siniflar arasindaki ayrimi onemser ve maksimize etemeye calisir.PCA unsupervised, LDA supervised ozelliktedir.
#PCA veri hangi sinifa ait old. onemsizdir fakat LDA ise verinin hangi sinifa ait old. onemlidir.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train) #y_train siniflari ogrenmesi icin PCA farkli olarak ekliyoruz.
X_test = lda.transform(X_test) #Ogrendigi icin y_test gerek yok.

#PCA siniflara bakmadan yaptigi icin veri kaybindan dolayi hata olurken.LDA siniflar gozeterek yaptigi icin hata olmadan yapabiliyor.

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(yv,df3,test_size=0.30, random_state=0) #Egitim ve test verilerini ayiriyoruz.
#random_state ozelligi np.random.seed benzer sekilde ayni rassalligin olmasini saglamak icin.
#test_size ise egitim ve test verilerinin hangi oranda ayrilacagini belirtmek icin. Bu ornek %70 Egitim , %30 Test ayrilmistir.
X_train, X_test, y_train, y_test = train_test_split(X,                 y, test_size=0.3, random_state=0)
#                                                   X :oznitelikler   ,y: label
#Bunlara bagimli bagimsiz degiskenler de denilebilir.

#Olceklendirme
""
MinMaxScaler:
Verileri 0-1 araligina indirilmeisi icin kullanilir.Veriler birbirlerine olan mesafe oranlari korunur.
Formul = x - min(veri) / max(veri) - min(veri) 

Standard Scaler:
scaler yontemi icin veriler normal dagildigi varsayilarak verilerin ortalama degeri= 0 ve std = 1 olacak sekilde verileri yeniden olcekler.
Veriler normal dagilmiyor ise bu yontemi kullanmak hatadir.
Formul = x - ortalama / standart sapma

RobustScaler:
Min-Max farki outlier == aykırı veya marjinal verilere karsi robust == dayanikli olmasidir.

Normalizer:
Veriler cok boyutlu ise tek formulle olceklemek icin kullanilir.


See Also : https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
"""
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()

from sklearn.preprocessing import minmax_scale

from sklearn.preprocessing import MaxAbsScaler
transformer = MaxAbsScaler().fit(X)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)

from sklearn.preprocessing import Normalizer
transformer = Normalizer().fit(X)

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(n_quantiles=10, random_state=0)

from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)



import statsmodels.api as sm
X = np.append(arr = np.ones((22,1)).astype(int), values=veri, axis=1)
X_l = veri.iloc[:,[0,1,2,3]].values 
X_l = np.array(X_l,dtype=float)
model = sm.OLS(veri.cinsiyetX_l).fit()
print(model.summary())
#En yuksek p-value degeri cikartilip tekrar test edilerek Backward Elimination yapilir.
#Fakat bazi durumlarda p-value degeri yuksek oldugu halde model basarisi icin gerekli olabiliyor bu yuzden kalmasi da degerlendir.
X_l = veri.iloc[:,[1,2,3]].values #En yuksek ilk ozellik ciktigindan bu ozellik cikartilip tekrar denenir.
X_l = np.array(X_l,dtype=float)
model = sm.OLS(veri.cinsiyet,X_l).fit()
print(model.summary())
#Makine ogrenmesi modellerine gore
lin_reg = LinearRegression()
lin_reg.fit(X,Y) #fit egitim surecidir.
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())
print("Linear R-Square degeri:")
print(r2_score(Y, lin_reg.predict((X))))


#Lineer Regresyon
"""
y = ax + b
y: bagimli degisken
x: bagimsiz degisken
"""


from sklearn.linear_model import LinearRegression

"""
Tum makine ogrenmesi methodlarinda bazilari olmasa da genel olarak methodlar:
Methods

decision_function(X)
decision_path(X)
apply(X)
fit(X[, y, sample_weight])
fit_predict(X[, y, sample_weight])
fit_transform(X[, y, sample_weight])
get_feature_names_out([input_features])
get_params([deep])
predict(X[, sample_weight])
predict_log_proba(X)
predict_proba(X)
score(X[, y, sample_weight])
set_output(*[, transform])
set_params(**params)
transform(X)
cost_complexity_pruning_path(X, y[, ...])
get_depth()
get_n_leaves()
"""


#Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron
#Polynom degree degistirdigimizde basarim artabilir.Veriler egrimize daha yakin olabilir.
 
#SVR
from sklearn import svm
#SVR ve SVM icin Standart Scaler kullanmak gerekir cunku aykiri ve marjinal degerlerden cok fazla etkilenirler.

#Decision Trees
from sklearn import tree


#Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
#Ensemble Learning == Kolektif Ogrenme ile birden fazla model ayni anda birlikte kullanilarak ogrenilmesini saglayan ogrenmedir.
#Majority Vote == Cogunlugun oyu siniflandirmada en cok hangisi ise o sinifa dahil olurken.Regression ortalamalari aliniyor.
#Birden fazla karar agaci modeli kullaniliyor.
clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
#n_estimators parametresi ile kac tane karar agaci olusturacagini belirtiyoruz.
"""
Parametreler:
n_estimatorsint, default=100
criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
bootstrapbool, default=True
oob_scorebool, default=False
n_jobsint, default=None
warm_startbool, default=False
random_stateint, RandomState instance or None, default=None
verboseint, default=0

ve karar agicanda olan parametrelerde mevcuttur.
"""

#Logistic Regression

#K-NN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, metric="manhattan")
#n_neighborsint degerinin default olarak 5 verilmistir.
#k=n yani ornek sayisina esitlediginde hangisi cogunlukta ise o sinifa dahil etcek tamamini
#Voronoi hucreleri verdigimiz de k degerine gore hucrelere ayiriyor ve hangi hucreye deger geldi ise o sinifa dahil ediyor.
#k icin Pattern Classification kitabinda bir formul verilmistir.
#sqrt(len(X_train))/2 == karekok(egitim boyutu)/ 2

"""
weights{‘uniform’, ‘distance’}, callable or None, default=’uniform’
Weight function used in prediction. Possible values:
"uniform" : uniform weights. All points in each neighborhood are weighted equally.
"distance" : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
[callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.

The valid distance metrics, and the function they map to, are:
metric                  Function
"cityblock"             metrics.pairwise.manhattan_distances
"cosine"                metrics.pairwise.cosine_distances
"euclidean"             metrics.pairwise.euclidean_distances
"haversine"             metrics.pairwise.haversine_distances
"l1"                    metrics.pairwise.manhattan_distances
"l2"                    metrics.pairwise.euclidean_distances
"manhattan"             metrics.pairwise.manhattan_distances
"nan_euclidean"         metrics.pairwise.nan_euclidean_distances
'chebyshev', 'cityblock', 'euclidean', 'infinity', 'l1', 'l2', 'manhattan', 'minkowski', 'p'

algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
Algorithm used to compute the nearest neighbors:
"ball_tree" will use BallTree
"kd_tree" will use KDTree
"brute" will use a brute-force search.
"auto" will attempt to decide the most appropriate algorithm based on the values passed to fit method.
"""

#SVM
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
rbf_svc = svm.SVC(kernel='rbf')
"""
Kernel functions:
"linear"
"polynomial"
"rbf"
"sigmoid"
"precomputed"
""

#Kendimiz Custom Kernel olusturup kullanabiliriz.
def my_kernel(X, Y):
    return np.dot(X, Y.T)

my_svc = svm.SVC(kernel=my_kernel)

#Diger parametreleri
Cfloat, default=1.0
degreeint, default=3

gamma{‘scale’, ‘auto’} or float, default=’scale’
Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
if ‘auto’, uses 1 / n_features
if float, must be non-negative.

coef0float, default=0.0  Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
probabilitybool, default=False
tolfloat, default=1e-3    Tolerance for stopping criterion.
decision_function_shape{‘ovo’, ‘ovr’}, default=’ovr’
break_tiesbool, default=False

"""
"""
#Hard marjin icine hic veri kabul etmezken.Soft marjin icine veri kabul edebiliyor.
#Kernel Trick == Cekirdek hilesi ile Non-linear == dogrusal olmayan verilerde boyut arttirarak verileri ayirmak icin kullanilir.
Boyut arttirmamizin nedeni verileri ayristirmayi kolaylastirmak icin.
"""

#Naive Bayes
"""
B kosulu altinda A'nin gerceklesme olasiligi
#P(A|B) = P(B|A) * P(A) / P(B)
Bayes' theorem may be derived from the definition of conditional probability:
#P(A|B) = P(AnB) / P(B)
"""
"""
Gaussian Naive Bayes : Surekli degerler icin kullanilan yontemdir.
Multinomial Naive Bayes : Multinomial degerler icin.
Complement Naive Bayes : It is particularly suited for imbalanced data sets.
Bernoulli Naive Bayes : binary-valued (Bernoulli, boolean) variable yani ikili degerler 1 ve 0 gibi ise kullanilir.
Categorical Naive Bayes : categorical naive Bayes algorithm for categorically distributed data 
Out-of-core naive Bayes model fitting :
"""
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(force_alpha=True)

from sklearn.naive_bayes import ComplementNB
clf = ComplementNB(force_alpha=True)

from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB(force_alpha=True)

from sklearn.naive_bayes import CategoricalNB
clf = CategoricalNB(force_alpha=True)

clf.fit(X, Y)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = "entropy")
"""
Parametreler:
criterion{"gini", "entropy", "log_loss"}, default=”gini”
splitter{“best”, “random”}, default=”best”
max_depthint, default=None
min_samples_splitint or float, default=2
min_samples_leafint or float, default=1
min_weight_fraction_leaffloat, default=0.0
max_featuresint, float or {“auto”, “sqrt”, “log2”}, default=None
random_stateint, RandomState instance or None, default=None
max_leaf_nodesint, default=None
min_impurity_decreasefloat, default=0.0
class_weightdict, list of dict or “balanced”, default=None
ccp_alphanon-negative float, default=0.0
"""
clf = clf.fit(X, Y)

#K-Means
#X-Means k degerini kendisi buldugu alternatif bir algoritmadir.
#WSCC (within-cluster sum of square) ile k degerini belirlemek icin kullaniyoruz.
#k degerini secmek icin kirilma noktasi == elbow point ile belirlenir.
#K-Means++ baslangic noktalarinin rastgeleligindeki problemi onlemek icin gelistirilmis bir algoritmadir.
#Bu rastgelelikten gelen probleme Baslangic Noktasi Tuzagi deniyor.
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X) #n_clusters kac kume olacagini beilirtmek icin k degeridir.
kmeans = KMeans(n_clusters=2, n_init="auto", init = "k-means++").fit(X)
print(kmeans.cluster_centers_) #Kumelerin merkezlerinin nerede oldugunun kordinatlarini veriyor.

veri = pd.read_csv("https://bilkav.com/musteriler.csv")
X = veri.iloc[:,3:]
ks = list()
for i in range(1,20): #n_clusters parametresi 0 degerini kabul etmediginden range 1 ile baslatmalisin.
  kmeans = KMeans(n_clusters=i, n_init="auto", init = "k-means++", random_state=0).fit(X)
  ks.append(kmeans.inertia_) #kmeans.inertia_  WSCC degerlerini bize veriyor.

plt.plot(range(1,20), ks) #WSCC degerini dusurmek icin kirilimlarda en dusukte olani secebiliriz bariz bir kirilim yoksa
#Diger kirilim noktalari da secilebilir basarim bu kirilim noktalari ile denenebilir.

"""
Parametreler:
n_clustersint, default=8
init{"k-means++", "random"}, callable or array-like of shape (n_clusters, n_features), default="k-means++"
max_iterint, default=300
tolfloat, default=1e-4
verboseint, default=0
random_stateint, RandomState instance or None, default=None
copy_xbool, default=True
algorithm{"lloyd", "elkan", "auto", "full"}, default="lloyd"
"""

#Hiyerarsik bolutleme
#Dendogram aradaki mesafefinin en uzak oldugu nokta ile yine kac kume olmasi gerektigini bulabiliriz.
#Ward's Method veya Ward mesafesi WSCC kullanarak iki kume arasindaki mesafenin olcumu icin kullaniyoruz.
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters = 3, affinity ="euclidean",linkage= "ward").fit(X)
#Not:  If linkage is “ward”, only “euclidean” is accepted.


"""
Parametreler:
n_clustersint or None, default=2
affinitystr or callable, default=’euclidean’
metricstr or callable, default=None
memorystr or object with the joblib.Memory interface, default=None
connectivityarray-like or callable, default=None
compute_full_tree‘auto’ or bool, default=’auto’
linkage{"ward", "complete", "average", "single"}, default="ward"
distance_thresholdfloat, default=None
compute_distancesbool, default=False
"""

import scipy.cluster.hierarchy as hierc
dendrogram = hierc.dendrogram(hierc.linkage(X, method="ward"))
plt.show() #Dendogram gormek icin.

#Association rule learning
#Apriori algorithm
#Eclat algorithm
#Apriori uses breadth-first search (BFS), Eclat depth-first search (DFS)
#Eclat ve Apriori scikit-learn kutuphanesinde yoktur.Hazir olarak github kodlara bakilabilir veya kendiniz yazabilirsiniz.
#Association rule mining algoritmalarinin oldugu WEKA programini kullanabilirsiniz.

#UCB (Upper Confidence Bound) == Ust Guven Siniri

#XGBoost
from xgboost import XGBClassifier
xbst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
xbst.fit(X_train, y_train)
y_pred = xbst.predict(X_test)

#LabelPropagation
from sklearn.semi_supervised import LabelPropagation

#LabelSpreading
from sklearn.semi_supervised import LabelSpreading

#Modellerin Dogrulugunun Olculmesi ve Model Secimi
#Evaluation of Predictions
"""
R-Square (R^2)
Hata Kareleri Toplami = E(yi - y'i)^2
Ortalama Farklarin Toplami = E(yi - yort)^2
R^2 = 1 - HKT / OFT
R^2 = 0 oldugu durum tum tahminlerin tamamina ortalama degeri verildiginde cikar.Bu en kotu durumdur.
HKT = 0 oldugunda R^2 = 1-0 => 1 gelir.Bu en iyi durumdur.
R^2 hatta negatif cikiyorsa tahminin cok kotu yapildigini belirtir.
Yani 1'e yakin olmali 0'a yakin veya altinda bir deger kotu oldugunu belirtir.
R^2 bazi durumlarda yetersiz kalir olumsuzlugun hangi degiskenden kaldigini gostermez bu yuzden farkli yontemlerde kullanilir.
"""
"""
Adjusted R-Square
Adjusted R^2 = 1 - (1-R^2) * n-1 / n-p-1
"""
from sklearn.metrics import r2_score
r2_score(y_true, y_pred)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred)
"""
Ac \ Pre    True(PP)  False(PN)
True (P)        TP    FN
False(N)        FP    TN

FP (False Positive) : Tip 1 Hata
FN (False Negative) : Tip 2 Hata

Accuracy = (TP + TN) / (total or all)
Specifity = TN / actual no
Error Rate = 1 - Accuracy
True Positive Rate = TP / actual yes
False Positive Rate = FP / actual no
Precision = TP / predicted yes
Prevalence = actual yes / total
....
https://en.wikipedia.org/wiki/Confusion_matrix
"""
#ZeroR algorithm
#ROC Egrisi
#TPR ne kadar yakinsa o kadar basalidir.FPR ne kadar yakinsa o kadar hatalidir.Fakat FPR TPR tarafina cevirmek basittir.
#FPR yanlis siniflandirilmis veya egitilmistir degerleri 1 0 yer degistirildiginde sorun cozulur.
#ROC egrisinin altinda kalan alanda ise bu algoritmalarin kullanimina gerek yoktur.
#AUC egriside bize ne kadar egitildiginde diger algoritmalara gore basari sagliyacagini gosterir.
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
metrics.auc(fpr, tpr)
roc_auc_score(y, clf.predict_proba(X)[:, 1])
roc_auc_score(y, clf.predict_proba(X), multi_class='ovr') #Multiclass
roc_auc_score(y, y_pred, average=None) #Multilabel
y_proba = ml_model.predict_proba(X_test)
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)

#k-katlamali capraz dogrulama
from sklearn.model_selection import cross_val_score
cvscr =  cross_val_score(model, X_train, y_train, cv=4)
#Ilk parametre modelimizi ornek clf , bagimsiz, bagimli , cv kac kere katliyacagimizi belirtiyoruz. 
print(cvscr.mean()) #Ortalama ne kadar yuksek ise o kadar basarilidir.  
print(cvscr.std()) #STD degerimiz az olmalidir.
#Sadece bu iki yontem ile degil max , min , ... gibi bircok kriterler ile basariyi gorebiliriz.

#GridSearchCV
from sklearn.model_selection import GridSearchCV
#GridSearch CV -> cross_val_score ozelliklerine gore yapabiliyor.
#GridSearchCV ile model secimi ve modellerin parametrelerinin de ne olacagini bulmak icin kullanilir.
params = [
            {'kernel':('linear'), 'C':[1,2,3,4,5]},
            {'kernel':('rbf'), 'C':[1,2,3,4,5]}          
         ]
gds = GridSearchCV(model, params, scoring= "accuracy", cv=4, n_jobs = -1)
#model kismina ml modelimizi, params parametresi ile denemek istedigimiz parametreleri, scoring neye gore skorlayacagini belirtir
#n_job ise paralelde kac is yapilacagini belirtmek icin girilen degerlerdir.
grid_search = gds.fit(X_train,y_train)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_index_)

"""
The scoring parameters:
"accuracy"          metrics.accuracy_score
"balanced_accuracy" metrics.balanced_accuracy_score
"top_k_accuracy"    metrics.top_k_accuracy_score
"average_precision" metrics.average_precision_score
"neg_brier_score"   metrics.brier_score_loss
"f1"                metrics.f1_score
"f1_micro"              ||
"f1_macro"              ||
"f1_weighted"           ||
"f1_samples"        metrics.f1_score
"neg_log_loss"      metrics.log_loss
"precision"         metrics.precision_score
"recall"            metrics.recall_score
"jaccard"           metrics.jaccard_score
"roc_auc"           metrics.roc_auc_score
"roc_auc_ovr"       metrics.roc_auc_score
"roc_auc_ovo"       metrics.roc_auc_score
"roc_auc_ovr_weighted"    metrics.roc_auc_score
"roc_auc_ovo_weighted"    metrics.roc_auc_score

Clustering

"adjusted_mutual_info_score"  metrics.adjusted_mutual_info_score
"adjusted_rand_score"         metrics.adjusted_rand_score
"completeness_score"          metrics.completeness_score
"fowlkes_mallows_score"       metrics.fowlkes_mallows_score
"homogeneity_score"           metrics.homogeneity_score
"mutual_info_score"           metrics.mutual_info_score
"normalized_mutual_info_score"  metrics.normalized_mutual_info_score
"rand_score"                    metrics.rand_score
"v_measure_score"               metrics.v_measure_score
	
Regression
		
"explained_variance"            metrics.explained_variance_score
"max_error"                     metrics.max_error
"neg_mean_absolute_error"       metrics.mean_absolute_error
"neg_mean_squared_error"        metrics.mean_squared_error
"neg_root_mean_squared_error"   metrics.mean_squared_error
"neg_mean_squared_log_error"    metrics.mean_squared_log_error
"neg_median_absolute_error"     metrics.median_absolute_error
"r2"                            metrics.r2_score
"neg_mean_poisson_deviance"     metrics.mean_poisson_deviance
"neg_mean_gamma_deviance"       metrics.mean_gamma_deviance
"neg_mean_absolute_percentage_error"    metrics.mean_absolute_percentage_error
"d2_absolute_error_score"               metrics.d2_absolute_error_score
"d2_pinball_score"                      metrics.d2_pinball_score
"d2_tweedie_score"                      metrics.d2_tweedie_score
	

"""

#Deep learning (DL) == Derin Ogrenme
#Artificial neural network (ANN) == Yapay Sinir Aglari
#I\O -> IPO : Input Process Output
#Standardize (Standartlastirilmis) 0-1 aralinda veriler olmalidir.
#Girdiler bagimsiz degiskenler ve ciktilar bagimli degiskenlerdir.
#Girdi 0-1 araliginda oldugu gibi Cikti 0-1 araliginda olur.
#Cikti 0-1 fakat binomial ve kategorik olarak da olabilir.
#Bazi normalizasyonlarda -1 ile 1 araliginda da olabilir.
#Girdi ve ciktilar cok olabilir.Ayni zamanda girdi ve ciktilarda yine norondur.
#Giriler Input Layer == girdi katmani'dir.
#Ciktilar Output Layer == cikti katmani'dir.
#Aradaki noronlarda Hidden Layer == gizli katman'dir.
#Synapsis agirliklar ile birlikte degiskenler noronlara tasinir.Noronda bu sinyaller toplanir.
#Noron uzerindeki sinyal = w1*x1 + w2*x2 + w3*x3 + ...
# y = w * x + b
#w : agirlik, b : bias, x : girisler
#Bias shift == kaydirma islemi icin kullaniliyor.
#Neuron calismasi icin aktivasyon fonksiyonuna ihtiyac vardir.
#Aktivasyon fonksiyonu olarak aslinda normal f(x) olarak tanimladigimiz fonksiyonlarda olabilir.
#Yani kendimiz ozel bir amac icin aktivasyon fonksiyonu belirleyip bunu uygulayabiliriz.
#Vanishing Gradients == Gradyanlarin yok olmasi problemi turevin sifir oldugu yerler ogrenme olmamasi problemidir.
#Exploding Gradients == Gradyanlarin Patlamasi problemi turevi ustel olarak artmasi problemidir.
#Gradyan inis problemlerine cozumlerden bazilari GRU, LSTM, BRNN,Deep RNN'dir.
#Gated Recurrent Units (GRU) == Gecitlenmis Ozyinelemeli Birimler , Vanishing Gradients problemini cozmek icin bir onceki degeri tutarak sagliyor.
#Full GRU daha cok kullanilan yontemlerden biri yine GRU gore bu yontemde ilgililik geciti ekleniyor. 
#Long Short Term Memory (LSTM) , GRU formullerde farklilik olan ve unutma geciti ve cikis bulunun bir versiyonu diebiliriz.
#Bidirectional RNN (BRNN) == Cift Yonlu Ozyinelemeli Sinir Aglari ile hem gecmisteki hemde gelecekteki degerler ile islem yapilir
#Capsule Network == Kapsul Aglar, CNN pooling isleminin hatali sonuclar vermesinden dogmustur.
#Evrisim isleminden ayiran ozelligi squashing function == ezme aktivasyon fonksiyonudur.
#Capsule Network girdi ve cikti vector'dur.Vectorel hesaplama yapilir.Standart NN'lerde cikislar skalerken Capsule'de vectoreldir
#Generative Adversarial Network (GAN) ==  Uretici Cekismeli Aglar 

model.coef_ #Katsayilari gormek icin
model.intercept_ #Sabiti gormek icin

"""
Yapay Sinir Aginin Ogrenmesi:

Feed Forward Propagation == Ileri Yayilim Algoritmasi
Z1 = w1 * x + b1
A1 = g1 * Z1
Z2 = w3 * A1 + b2
A2 = g2 (Z2)
g: aktivasyon fonksiyonu, w : agirlik, b: bias


Back Propagation == Geriye Yayilim Algoritmasi
dZ2 = A2-Y
dW2 = 1/m DZ3 - A1
dZ1 = W2T dZ2 * g1Z1
      (n1,m) ,  (n1,m)

dW1 = 1/m dZ1 XT

#Back Propagation yontemi ile turev alinarak agirlik degerlerini guncelliyoruz.
"""


"""
Activation Functions:
Step Function == Adim Fonksiyonu or Threshold Function == Esik Fonksiyonu
Sigmoid Function
Rectifier Function == Duzlestirilmis Fonksiyon
Hyperbolic Tangent (tanh) == Hiperbolik Tanjant
...
"""
#YSA farkli yontemlerle ogrenebilir bunlardan biri perceptron'dur.
#Learning rate ne kadar hizli ogrenecegini belirtmek icin genelde bu deger onceden verilir.
#Gradient Descendent ile en optimum nokta bulunarak hatalar duzeltilir.
#Stochastic Gradient Descendent ile her veride yapilirken Batch yaklasimi tum veriyi okuyup agirliklari duzelettigimiz yontemler.
#Mini Batch veriyi parcalayip parcaladigi verileri agirliklari duzelettigimiz yontem.
#Epoch ile kac kere turlayacagini belirtebiliriz.
#Epoch degeri az verilirse ogrenmemis, cok verilirse de gereksiz calismis olur.Bu yuzden epoch degerini belirlemek icin yontemler gelistirilmistir.
#Back Propagation == Geri yayilim, Forward Propagation == Ileri yayilim
#Veri on isleme islemini yeterli noron verildiginde kendisi yapiyor.Fakat bazi akademik gorusler yine de on isleme yapilmali der.
#Fakat bizler tum ozniteliklerimiz girdi olarak verelim.Sistem kendisi duzenliyecektir.On isleme yapsakta.
#Dropout Layer yontemi ile genelde %20-%30 secilirken max %50 olabilir.Bazi gizli katmandaki noronlari cikaririz.Bu basarimi arttirabilir.
#Bias Variance iliskisinde ise optimum nokta belirlenmelidir.Bias yuksek iken model egtiminde basarili iken test basarisiz.
#Variance yuksek iken test basarili iken model egitiminde basarisizdir.Bu yuzden en optimum min nokta belirlenmeldir.
#Egitim ve test verileri ayiriken verimiz cok buyuk ise egitim icin %98 , %1 validation, %1 test olabilirken.
#Veri setimiz az ise egtim %80'den buyuk olmamali ve test ve validation arttirilmalidir.
#Transfer Learning ogrenilmis kullanmaktir.
#Finetuning == Model Uyarlama daha buyuk veri setinden ogrenileni daha kucuk veri setinde uygulama islemidir.
#Multi-Task Learning bir birini iceren destekleyen sistemlerde birden fazla gorevi ayni anda ogrenebilir.
#Tek sinir agi var ise sig ag , cok fazla gizli katman var ise deep neural network == derin sinir ag
"""
L : Katman sayisi 
n^[l] : l. katmandaki noron sayisi

* : Evrisim islemi , carpma islemi degildir.
"""

"'"
Dense-Sparse-Dense Training yontemi ile ilk yogun katman ve noron ile hangi noronlarin seyreltilmesi gerektigi tespit edilir.
Daha sonra seyrek katman ve noronlar ile model egitilir.
Daha sonra tekrar yogun katman ve noronlar ile test edilir.Bu sayede basari arttirilabilir.
""
#Early stopping ile erken durdurarak model egitimi tamamlanir.Bu sayede overfitting == asiri ogrenmenin onune gecilir.



import keras
from keras.models import Sequential #Sequential model icin
from keras.layers import Dense #Layer olusturmak icin

clf = Sequential() #Sequential model ANN olusturmak icin.
clf.add(Dense(units= 4, activation= "relu", input_dim= 6)) #Input Layer
clf.add(Dropout(0.5)) #Seyreltme islemi icin.
clf.add(Dense(units= 4, activation= "relu")) #Hidden Layer
clf.add(Dropout(0.5))
clf.add(Dense(units= 1, activation= "sigmoid")) #Output Layer
#units : kac tane gizli katman oldugunu belirmek icin. activation hangi aktivasyon fonsiyonunu kullanmak istedigin.
#input_dim kac tane girdi noronu olacagini belirtiyoruz.Bunu bagimsiz degisken sayina esitlemek en dogrusudur.
#Cikis katmaninin aktivasyon fonksiyonu sigmoid olmasi onerilir.

#Yukaridak kod blogunu tek blokta da yapabiliriz.
clf = Sequential(
    [
        Dense(4, activation="relu", name="inputlayer"),
        Dense(4, activation="relu", name="hiddenlayer"),
        Dense(1, activation= "sigmoid", name="outputlayer"),
    ]
)


clf.compile(loss='binary_crossentropy', optimizer='adam', metrics= ["accuracy"])
#optimizer ile nasil optimize edilecegini belirtiyoruz.Synpapsislerdeki degerlerin nasil optimize edilecegini belirtiyoruz.
#loss kayiplar icin.
#metrics ile neyi optimize edecegini belirtiyoruz. ["accuracy"] yaparsak accuracy'i arttirmaya calisacak.

clf.fit(X_train, y_train, epochs=100)
y_pred = clf.predict(X_test)

"""
Core layers:
Input object
Dense layer
Activation layer
Embedding layer
Masking layer
Lambda layer

Convolution layers, Pooling layers, Recurrent layers, Preprocessing layers, Normalization layers, Regularization layers,
Attention layers, Reshaping layers, Merging layers, Locally-connected layers, Activation layers
Core Layers haricindedir ve bunlarin altinda layers mevcuttur.

Soruce : https://keras.io/api/layers/
"""
"""
Dense Layer Arguments:
units: Positive integer, dimensionality of the output space.
activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
use_bias: Boolean, whether the layer uses a bias vector.
kernel_initializer: Initializer for the kernel weights matrix.
bias_initializer: Initializer for the bias vector.
kernel_regularizer: Regularizer function applied to the kernel weights matrix.
bias_regularizer: Regularizer function applied to the bias vector.
activity_regularizer: Regularizer function applied to the output of the layer (its "activation").
kernel_constraint: Constraint function applied to the kernel weights matrix.
bias_constraint: Constraint function applied to the bias vector.

Source: https://keras.io/api/layers/core_layers/dense/
"""
"""
Aktivasyon Fonksiyonlari:
relu function
sigmoid function
softmax function
softplus function
softsign function
tanh function
selu function
elu function
exponential function

Source: https://keras.io/api/layers/activations/
"""

"""
Available optimizers:
SGD
RMSprop
Adam
AdamW
Adadelta
Adagrad
Adamax
Adafactor
Nadam
Ftrl

Source : https://keras.io/api/optimizers/
"""

#Convolution Neural Network (CNN) == Evrisimsel Sinir Aglari
#CNN farkliliklari bulmak uzerine kuruludur.
#CNN Image Processing == Goruntu Isleme icin kullanilan yontemlerden bir tanesidir.
#CNN oznitelikleri cikartip belli bir sinifa dahil edebilen yontemdir.
#Yani CNN oznetelik cikarimi, preprocessing islemleri, makine ogrenmesi modellerinin cikartilmasi adimlarini ustleniyor.
#Flattening == Duzlestirme isleminden once Convolution ve Pooling istenildigi kadar tekara edilebilir.
#Flattening isleminden sonra tam bagli ileri beslemeli ysa input olarak gider.
#Pooling isleminde en cok kullanilan yontem Max Poolingdir.Bu islem cercevdeki max pixel alip resmi kucultme islemidir.
#Convolution ise bir filtre veya donusum islemidir.
#Flattening islemi gelen 2 boyutlu matrisi tek boyuta donusturuyor.Satir satir veya stun bazli olarak sirasiyla matris alinip vektore ceviriyor.
#Pooling yerine downsampling terimi de kullanilabiliyor.
#See also : https://adamharley.com/nn_vis/cnn/2d.html
#relu CNN ve goruntu isleme icin en cok kullanilan aktivasyon fonksiyonudur.

#Recurrent Neural Network (RNN) == Yinelemeli Sinir Aglari
#RNN en cok kullanilan aktivasyon fonksiyonu relu cok kullanilmiyor onun yerine tanh kullanilir.
#RNN cikisi icin yine sigmoid kullanilir.Softmax'te kullanilabiliyor.
#Long Sort Term Memory (LSTM) == Uzun kısa süreli bellek  bir tur RNN'dir.

#Automated machine learning (AutoML)
#Auto-Keras, Auto-PyTorch, Auto-Sklearn, AutoGluon, H2O AutoML ,AutoML, TPOT araclari ile ml otomatik hale getirilebilir.
#IBM ve Google AutoML cozumleri mevcuttur.

#AutoKeras
pip3 install autokeras
import autokeras as ak

clf = ak.ImageClassifier()
clf.fit(x_train, y_train)
results = clf.predict(x_test)
#AutoKeras Image ve Text icin kullanilabiliyor.

#Structured Data Classification
clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3)
clf.fit(x_train, y_train, epochs=10)
y_pred = clf.predict(x_test)
print(clf.evaluate(x_test, y_test))

#Structured Data Regression
reg = ak.StructuredDataRegressor(max_trials=3, overwrite=True)
reg.fit(x_train, y_train, epochs=10)
y_pred = reg.predict(x_test)
print(reg.evaluate(x_test, y_test))

#Auto-sklearn
pip install auto-sklearn
import autosklearn.classification
clf = autosklearn.classification.AutoSklearnClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

import autosklearn.classification
automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)

from autosklearn.experimental.askl2 import AutoSklearn2Classifier #2.0 cikmis

#Egitilmis Modelin Tekrar Kullanilmasi
model.save("file_name")#Keras icin disayia kaydetme.Farkli kodlarda vardi sonra bak : https://www.youtube.com/watch?v=YCXFceVKHTk
#Pickle == Tursu, joblib,Pmml kutuphaneleri mevcuttur.Pmml bir standar saglar diger dillere egitilmis veri aktarilabilir.
#Pickle Python standart kutuphanelerinden biridir.
#https://docs.python.org/3/library/pickle.html?highlight=pickle#module-pickle
import pickle
pickle.dump(model, open("file_name", "wb")) #dump ile onceden egitilmis modelimizi bir dosyaya kaydetmek icin kullaniyoruz.
#open ile dosyayi acip yazma islemi icin w, binary olarak kaydetmesi icin b degerlerini veriyoruz.
ogrenilmis = pickle.load(open("file_name", "rb")) #load ile dosyaya kaydetmis oldugumuz egitilmis modeli yuklemek icin.
ogrenilmis.predict(X_test) #Artik nasil islemler yapmak isterseniz ogrenilmis modeli kullanabilirsiniz.

