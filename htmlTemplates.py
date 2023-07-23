css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://www.google.com/imgres?imgurl=https%3A%2F%2Fimg.etimg.com%2Fthumb%2Fmsid-100838863%2Cwidth-1200%2Cheight-900%2Cimgsize-46420%2Coverlay-economictimes%2Fphoto.jpg&tbnid=dVOTXkiXLbWCLM&vet=12ahUKEwi22rPO_52AAxWAlmMGHfdHA84QMygGegUIARD1AQ..i&imgrefurl=https%3A%2F%2Feconomictimes.indiatimes.com%2Fnews%2Findia%2Fpm-modi-to-address-indian-americans-in-washington-on-june-23-community-leader%2Farticleshow%2F100838702.cms&docid=PQX064cYKzNkxM&w=1200&h=900&q=narendra%20modi&ved=2ahUKEwi22rPO_52AAxWAlmMGHfdHA84QMygGegUIARD1AQ" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.google.com/imgres?imgurl=https%3A%2F%2Flookaside.fbsbx.com%2Flookaside%2Fcrawler%2Fmedia%2F%3Fmedia_id%3D100044149130762&tbnid=yHoZ1xbL6OYugM&vet=12ahUKEwjCxYno_52AAxUX5zgGHWKVBmsQMygEegUIARDqAQ..i&imgrefurl=https%3A%2F%2Fwww.facebook.com%2Famitshahofficial%2F&docid=nISFTI6YUEbGFM&w=1829&h=1829&q=amit%20shah&ved=2ahUKEwjCxYno_52AAxUX5zgGHWKVBmsQMygEegUIARDqAQ">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''