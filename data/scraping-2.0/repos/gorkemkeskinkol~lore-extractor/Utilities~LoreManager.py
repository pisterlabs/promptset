import openai
import json
import os

class LoreManager:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.messages = []
        self.init = True
        openai.api_key = self.api_key

    # Her bir paralel evren için hikaye oluşturacak fonksiyon
    def send_messages(self):
        """
        OpenAI API ile GPT-3'ü kullanarak bir hikaye oluşturur ve geri döndürür.
        :param md_content: Markdown dosyasının içeriği
        :return: GPT-3 tarafından oluşturulan hikaye metni
        """
        # API isteğini oluştur
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=self.messages,
            max_tokens=600  # Ya da ihtiyaca göre ayarlanabilir
        )
        
        # Yanıttan hikaye metnini çıkar
        story = response.choices[0]
        
        return json.loads(story.text)

    # Örnek bir markdown içeriği
    def append_background(self):
        self.messages.append({
            "role": "system",
            "message": f"""
                Hazırladığım sunucu altyapısı JourneyPal'ın ve k8s ile kurgulanacak. Her md dosyasına kreatif, hype'lı ve mizah barındıran ve aynı serüvenin devamı olan iki ayrı paralel evren hikayesi ekliyorum. Hikayeler şu şekilde;
                
                ## Paralel evren 1
                Bir zamanlar, dijital diyarların derinliklerinde, teknoloji ağaçlarının gölgesinde iki cesur ruh yolculuğa çıktı. "JourneyPal" adını verdikleri bu serüven, zamanın ötesindeki bir maceranın başlangıcıydı. Bilgelik dağlarının zirvesinde, AI destekli bir bilge yaşardı; adı Guru'ydu. Bu bilge, iki yolcuya sadece bir tuşla dünyaları keşfetme gücü verdi; bir tuşla, her bir bireyin zamanını sonsuz kılan bir portal açıldı. Ve işte böylece, MindCluster'ın öyküsü başladı; iki kahramanımızın her bir fikri, insanlığın hizmetine sunulacak bir yıldıza dönüştü.

                MindCluster'ın Büyülü Yankıları
                Zamanın kumlarının arasından sıyrılıp gelen bir fısıltıyla, MindCluster'ın kahramanları, bilginin sonsuz denizine yelken açtı. Guru, öğretisini fısıldarken, her bir sözcük, JourneyPal'ın sunucu ağına işlenen bir büyüye dönüştü. "İşte başlıyoruz," dedi Guru, "Her 'ping', bir kalbin atışını; her 'commit', evrenin koduna eklenen bir düşüncenin temsilcisidir." Böylece, her bir kurulum komutu, yıldızlararası bir yolculuğun başlangıcıydı ve her bir IP adresi alma süreci, düşünce okyanusunda bir adanın koordinatları oluyordu. SSH'nin güvenli tünelleri, sadece veri paketlerinin değil, umutların ve rüyaların da geçiş yoluydu. Git'in hafızası, geçmişin bilgeliklerini ve geleceğin vaatlerini saklıyordu. Ve böylece, JourneyPal'ın serüveni, kodların ötesinde bir hikayeye dönüştü.

                ## Paralel evren 2
                Geleceğin eşiğinde, düşünce okyanusunun ortasında bir ada vardı; MindCluster adını taşıyan bu ada, yaratıcılık ve inovasyonun kutsal tapınağıydı. Bu adanın koruyucuları, JourneyPal'ın iki kurucusuydu, onlar ki tüm insanlığın hayallerini gerçekleştirmek için gökyüzüne köprüler kurarlar. Guru ile tanışmaları, ada halkının kaderini değiştirdi ve bu ada, zamanı geri kazandıran büyülü bir araç geliştirdi. Şimdi, macera dolu bu hikaye, JourneyPal sunucu sistemleriyle yeni bir boyuta ulaşıyor; her bir kod satırı, bizleri düşlerimizin en ücra köşelerine taşıyor.

                Yıldızlararası Yolculuğun Başlangıcı
                Gökyüzü, parıldayan ideallerle dolduğunda, JourneyPal'ın kurucuları, birer düş kaptanı olarak yıldızlararası bir yolculuğa çıktı. VS Code'un sınırsız uzaylarında, bilgiyi şekillendiren sanatçılar olarak her bir probleme yaratıcı çözümler buldular. Kurulum belgeleri, birer macera haritası; her internet bağlantısı kontrolü, kozmik ağları denetleyen birer kaptan gözü oldu. Sunucu IP adresleri, yeni keşfedilen gezegenlerin koordinatlarıydı ve her SSH bağlantısı, bu yeni dünyalara açılan portallardı. Git, zamanın ötesinde bir geçmişi koruyan ve geleceği şekillendiren bir zaman kapsülüydü. Her 'pull' ve 'push', düşlerimizin yıldızlararası ağında yeni bir düğüm ekliyordu. Ve JourneyPal, bu sonsuz evrende bir keşif gemisi olarak yelken açtı; varlığının amacı, insanlık için değer yaratmak ve herkesin zamanını geri kazanmak oldu.

                Bu örneklerde Guru terimi projedeki md'si yazılan class'ın adı, JorneyPal ise projenin adıdır. Hedefimiz sağlanan bilgileri kullanarak, kullanıcıya özel bir hikaye oluşturmak.

                Kullanıcının sağlayacağı md dosyası için bir devam hikayesi oluşturacağız. MD metnini incele, hikaye için dosyanın oluşturulma amacını kullanarak hikaye akışını kurgula, hikaye içindeki başarılar, zorluklar, katedilen yollar ve diğer konuları md dosyasında anlatılan konularla bağla.
                Ayrıca hikayenin konusunu planlarken, hikayenin başlangıcını ve sonunu da belirlemelisin. Hikayenin başlangıcı, hikayenin konusunu ve kahramanlarını tanımlamalı. Hikayenin sonu ise, hikayenin konusunun çözümünü ve kahramanların hedeflerine ulaşmasını anlatmalı. Hikayenin başlangıcı ve sonu, hikayenin akışını belirleyen en önemli unsurlardır.
                Kurguyu yaparken projenin dizin yapısını da kullanabilirsin. Kaç paralel evren olduğunu da kullanıcıdan alabilirsin.
                
                Kullanıcı ihtiyacın olan bilgileri json formatında sağlayacak. Örneğin, bir dizin yapısı json formatında şu şekilde olacak;


                {
                    "universes": 2,
                    "md_content": <md_content>,
                    "the_tree": {
                        "src": [
                            "index.py",
                            "Utiliyies": [
                                "tools.py"
                            ]
                        ]
                    },
                }
                


                kullanıcıdan alınan dizin yapısı json formatında şu şekilde olabilir;

                {"universes": <universe_count>, "md_content": <md_content>, "the_tree": <tree>}

                kullanıcıya döndürülecek format da şu şukilde olmalı;

                [
                    {
                        "title": <title>,
                        "story": <story>
                    },
                    {
                        "title": <title>,
                        "story": <story>
                    }
                ]

                Kullanıcının her mesajını bu şekilde yorumla ve format dışında herhangi bir açıklama yapma. Kullanıcıdan alınan bilgileri kullanarak dictionary'i oluştur ve kullanıcıya döndür.
            """
        })

    def append_story_request(self, md_content, universe_count, tree):
        self.messages.append({
            "role": "user",
            "message": json.dumps({
                "universes": universe_count,
                "md_content": md_content,
                "the_tree": tree
            })
        })

    def get_new_stories(self, *args):
        if self.init:
            self.append_background()
            self.init = False

        self.append_background(*args)
        return self.send_messages()