import spacy
import zh_core_web_lg
import neuralcoref

nlp = zh_core_web_lg.load()
neuralcoref.add_to_pipe(nlp)

# summarizer 載入中文模型
from summarizer import Summarizer
from summarizer.text_processors.sentence_handler import SentenceHandler
from spacy.lang.zh import Chinese
from transformers import *

# Load model, model config and tokenizer via Transformers
modelName = "bert-base-chinese" # 可以換成自己常用的
custom_config = AutoConfig.from_pretrained(modelName)
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained(modelName)
custom_model = AutoModel.from_pretrained(modelName, config=custom_config)

model = Summarizer(
    custom_model=custom_model,
    custom_tokenizer=custom_tokenizer,
    sentence_handler = SentenceHandler(language=Chinese)
    )

body = "2022年6月1日,行政院晚間宣布啟動「台美21世紀貿易倡議」。(黃信維攝)美國商務部和貿易代表署日前接連發布申請《晶片與科學法》和2023年貿易政策議程及2022年度報告,其中前者法規被認為衝擊台灣「矽盾」,後者則提到台美21世紀貿易倡議。知情人士表示,該倡議形同「類雙邊貿易協定」,而美國鼓勵在其境內生產晶片,只是為了要有安全備援。2022年7月,美國國會通過《晶片與科學法》(CHIPS and Science Act),美國總統拜登(Joe Biden)簽署生效,該法旨在鼓勵半導體廠商回到美國境內產製。美國商務部2月28日發布申請補助規則,將於6月下旬開始收件,並預期包括貸款和其擔保在內的獎勵補助,不會超過企業資本支出的35%。根據《晶片與科學法》,美國準備總額約520億美元的補助預算,其中390億美元用於設廠補助。2022年12月,全球半導體龍體「台積電」為美國亞利桑那州鳳凰城晶圓廠,舉行首批機台設備到廠典禮,拜登更是親自出席參與。不過台積電在美設廠,加上美國通過《晶片與科學法》,引發台灣「矽盾」不保疑慮。鼓勵境內產晶片當備援「數字會說話」,知情人士直言,以台積電鳳凰城新廠投入400億美元來看,美國給予的補助大概只夠建2個廠。該人士強調,台積電早在2020年5月就宣布要在美國設廠,而拜登政府是2021年上台,意即在美設廠與《晶片與科學法》並無直接關係,且該法旨在美國要有安全備援,並非成為晶片生產基地。知情人士說,假設遇到自然災害或特殊情況,導致台積電暫時無法產製足夠晶片時,美國就能靠在其境內的產量撐個數月,並稱台積電赴美設廠也是回應美國客戶需求。至於2022年6月啟動的台美21世紀貿易倡議,知情人士表示,可把此倡議視為「類雙邊貿易協定」(類BTA)。該人士強調,此倡議是「活的協議」(living agreement),隨時可擴充內容,所以當美國準備觸及關稅議題時,也能展開談判並納入倡議中,且這是拜登上任後,首個類似BTA的協議。另外,印太經濟架構(IPEF)也是拜登政府重要貿易政策之一,但我國未獲邀加入,知情人士則稱,倡議內容與IPEF相似。21世紀倡議同「類BTA」美國貿易代表署(USTR)報告指出,美國和台灣有長期且活躍的貿易關係,基於這樣的歷史,美國與台灣在美國在台協會(AIT)及駐美台北經濟文化代表處(TECRO)主持下,啟動台美21世紀貿易倡議,並於2022年8月宣布展開正式協商。該倡議目的在於發展具體方案,深化美國和台灣的經貿關係,基於共享價值來提升共同的貿易優先要務,並促進創新和對勞工、企業具包容性的經濟成長。美國和台灣也針對在11個貿易領域達到高標準承諾及有意義的經濟成果,發展具雄心壯志的路線圖。(推薦閱讀:美中競爭委員會主席低調訪台》夏季前來台辦聽證會?外交部這樣回應)台美21世紀貿易倡議涵蓋農業、反貪、數位貿易、環境、良好的監管實踐、勞工、非市場政策及實踐、中小企業、標準規範、國有企業、貿易便利11個領域。據悉,該倡議協商有望很快傳出好消息。我國外交部發言人劉永健2日提到,倡議部分領域已有實質進展,凸顯雙方促進經貿夥伴關係的積極意願。"
result = model(body, num_sentences=3)
full = ''.join(result)
print(full) # 摘要出來的句子