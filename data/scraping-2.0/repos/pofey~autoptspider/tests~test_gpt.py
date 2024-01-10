from autoptspider.openai.openaimanager import OpenAIManager
import re
import html


def test_gpt():
    """
    利用GPT测试种子分类解析
    :return:
    """
    html_source = """
    <table border="0" cellspacing="0" cellpadding="10"><tr><td style='border: none; padding: 10px; background: red'>
<b><a href="messages.php"><font color="white">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;你有1條新短訊！點擊查看&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</font></a></b></td></tr></table></p><br /><table class="main" width="940" border="0" cellspacing="0" cellpadding="0"><tr><td class="embedded" ><div id="usercpnav"><ul id="usercpmenu" class="menu"><li><a href="usercp.php">&nbsp;&nbsp;設&nbsp;&nbsp;定&nbsp;&nbsp;首&nbsp;&nbsp;頁&nbsp;&nbsp;</a></li><li><a href="?action=personal">&nbsp;&nbsp;個&nbsp;&nbsp;人&nbsp;&nbsp;設&nbsp;&nbsp;定&nbsp;&nbsp;</a></li><li class=selected><a href="?action=tracker">&nbsp;&nbsp;網&nbsp;&nbsp;站&nbsp;&nbsp;設&nbsp;&nbsp;定&nbsp;&nbsp;</a></li><li><a href="?action=forum">&nbsp;&nbsp;論&nbsp;&nbsp;壇&nbsp;&nbsp;設&nbsp;&nbsp;定&nbsp;&nbsp;</a></li><li><a href="?action=security">&nbsp;&nbsp;安&nbsp;&nbsp;全&nbsp;&nbsp;設&nbsp;&nbsp;定&nbsp;&nbsp;</a></li><li><a href="?action=laboratory">實驗室</a></li></ul></div></td></tr></table>
<table border=0 cellspacing=0 cellpadding=5 width=940><form method=post action=usercp.php><input type=hidden name=action value=tracker><input type=hidden name=type value=save><tr><td width="1%" class="rowhead nowrap" valign="top" align="right">默認<br />分類</td><td width="99%" class="rowfollow" valign="top" align="left"><table><tr><td class=embedded align=left><b>類型</b></td></tr><tr><td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat401 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=401"><img src="pic/cattrans.gif" alt="Movie(電影)/SD" title="Movie(電影)/SD" style="background-image: url(pic/category/chd/scenetorrents/cht/moviesd.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat419 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=419"><img src="pic/cattrans.gif" alt="Movie(電影)/HD" title="Movie(電影)/HD" style="background-image: url(pic/category/chd/scenetorrents/cht/moviehd.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat420 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=420"><img src="pic/cattrans.gif" alt="Movie(電影)/DVDiSo" title="Movie(電影)/DVDiSo" style="background-image: url(pic/category/chd/scenetorrents/cht/moviedvd.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat421 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=421"><img src="pic/cattrans.gif" alt="Movie(電影)/Blu-Ray" title="Movie(電影)/Blu-Ray" style="background-image: url(pic/category/chd/scenetorrents/cht/moviebd.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat439 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=439"><img src="pic/cattrans.gif" alt="Movie(電影)/Remux" title="Movie(電影)/Remux" style="background-image: url(pic/category/chd/scenetorrents/cht/movieremux.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat403 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=403"><img src="pic/cattrans.gif" alt="TV Series(影劇/綜藝)/SD" title="TV Series(影劇/綜藝)/SD" style="background-image: url(pic/category/chd/scenetorrents/cht/tvsd.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat402 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=402"><img src="pic/cattrans.gif" alt="TV Series(影劇/綜藝)/HD" title="TV Series(影劇/綜藝)/HD" style="background-image: url(pic/category/chd/scenetorrents/cht/tvhd.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat435 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=435"><img src="pic/cattrans.gif" alt="TV Series(影劇/綜藝)/DVDiSo" title="TV Series(影劇/綜藝)/DVDiSo" style="background-image: url(pic/category/chd/scenetorrents/cht/tvdvd.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat438 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=438"><img src="pic/cattrans.gif" alt="TV Series(影劇/綜藝)/BD" title="TV Series(影劇/綜藝)/BD" style="background-image: url(pic/category/chd/scenetorrents/cht/tvbd.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat404 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=404"><img src="pic/cattrans.gif" alt="紀錄教育" title="紀錄教育" style="background-image: url(pic/category/chd/scenetorrents/cht/bbc.png);" /></a></td>
</tr><tr><td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat405 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=405"><img src="pic/cattrans.gif" alt="Anime(動畫)" title="Anime(動畫)" style="background-image: url(pic/category/chd/scenetorrents/cht/anime.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat406 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=406"><img src="pic/cattrans.gif" alt="MV(演唱)" title="MV(演唱)" style="background-image: url(pic/category/chd/scenetorrents/cht/mv.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat408 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=408"><img src="pic/cattrans.gif" alt="Music(AAC/ALAC)" title="Music(AAC/ALAC)" style="background-image: url(pic/category/chd/scenetorrents/cht/mp3.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat434 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=434"><img src="pic/cattrans.gif" alt="Music(無損)" title="Music(無損)" style="background-image: url(pic/category/chd/scenetorrents/cht/flac.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat407 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=407"><img src="pic/cattrans.gif" alt="Sports(運動)" title="Sports(運動)" style="background-image: url(pic/category/chd/scenetorrents/cht/sport.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat422 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=422"><img src="pic/cattrans.gif" alt="Software(軟體)" title="Software(軟體)" style="background-image: url(pic/category/chd/scenetorrents/cht/software.png);" /></a></td>
<td align=left class=bottom style="padding-bottom: 4px;padding-left: 7px"><input class=checkbox name=cat423 type="checkbox"  value='yes'><a href="torrents.php?allsec=1&amp;cat=423"><img src="pic/cattrans.gif" alt="PCGame(PC遊戲)" title="PCGame(PC遊戲)" style="background-image: url(pic/category/chd/scenetorrents/cht/pcgame.png);" /></a></td>
    """
    OpenAIManager.set_api_key("换自己的apikey")
    text = OpenAIManager.get_response([
        {
            "role": "system",
            "content": """
                你是一位专业的前端专家，非常擅长使用css选择器分析网页，制作爬虫工具。
                接下来我将给你一段网页源代码，你要分析出其中所有的分类信息，并返回给我一个json结构
                下面是代码：
                %s
                下面是你应该返回给我json数组的格式：
                [{
                    "id": int,   
                    "name":string,
                    "media_type": string    
                }]
                其中id是分类的编号，一个数字
                name是分类的名称，一般为中文
                media_type是分类的类型，比如电影，电视剧，动漫等，它的数据限定为：Movie/TV/Documentary/Anime/Music/Game/AV/Other
                """ % (html_source)
        },
    ])
    print(text)
