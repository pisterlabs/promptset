import openai

openai.api_key = "Enter your key here"

generative_conversation = """
  T: Good morning. I’m Doctor Rogers.
  P: Hello, Dr. Rogers
  T: Won't you have a chair? Now then, we have half an hour together, and I really don't
  know what we will be able to make of it but uh I hope we can make something of it.
  I'd be glad to know whatever concerns you.
  T: And... I don't know whether you're feeling a little uptight under these lights and
  all, but I think I'm feeling a little uptight, but I don't think that will last very long.
  And I have, not having met you before I, I don't have any idea what sort of
  concerns or issues you want to bring up, but I'd be glad to hear whatever you
  want to say.
  P: I'm not quite sure where to begin.
  T: Uhm, hmm.
  P: I feel like I do bad things that make me ashamed of myself.
  T: What you'd like to do is to feel more accepting toward yourself when you do things
  that you feel are wrong. Is that right?
  P: Right. And I feel like, I feel like...
  T: It sounds like a tough assignment. I sure wish I could give you the answer as to what you should do.
  P: I’m not, I guess, sure he can accept me since I don’t accept me.
  T: I see. It really cuts a little deeper. If he really knew you, would he, could he
  accept you?
  P: Right. And I have a feeling that you are just
  going to sit there and let me stew in it and I want more. I want you to
  help me get rid of my guilt feeling
  T: Mhm. And I guess I'd like to say, "No, I don't want to let you stew in your feelings,"
  but on the other hand, I, I also feel that this is the kind of thing that I
  couldn't possibly answer for you. But I sure as anything will try to help you work
  toward your own answer. I don't know whether that makes any sense to you, but I
  mean it. I guess I do catch the real deep puzzlement that you feel as to "What the hell shall I do? What can I do?"
  P: Yes
  T: Mhm. But you feel, really, that at times you’re acting in ways that are not in
  accord with your own inner standards.
  C30 Right. Well, I have a hopeless feeling. I mean, these are all the things that I sort of feel myself,
  and I feel uh - O.K., now what?
  T30 Mhm. You feel this is the conflict and it's just insoluble, and therefore it is hopeless,
  and here you look to me and I don't seem to give you any help and that uh-
  P: Yes, exactly.
  T: Hi.
  P: Hi Carl.
  T: Good to see ya.
  P: I’ve been looking forward to this moment.
  T: Well, I’m eager to know what’s what with you.
  P: Well, do you mean in terms of the past year or right now?
  T: Oh, right now, whatever your concerns are in your present life. I
  guess that’s what I’m interested in.
  P: Well, um, (smiles) I come here with an agenda, like, of a particular thing
  I’d like to discuss with you. And, and I think that discussing it will be
  helpful in terms of my, of my life.
  T: Uh-huh. Then that’s what I’d like you to discuss.
  P: Well, I think I’m doing very well in many ways, but I still need to do some work. But I’m not sure if I can do it and it’s very important.
  T: Are you afraid of the responsibility or, what aspect of it is most frightening?
  P: I’m not able to, to live my life. I feel like a disappointment and a failure and there’s nothing I can do about it.
  T: You just feel, “I may not be able to make it. I may be doomed to failure by the very circumstances.
  P : And talking about it this way I feel
  two ways: Like, “Oh c’mon. Everybody else is doing it too.” And on the
  other hand, like, you’re willing to sit here and talk about it with me and I appreciate that.
  T: Mhm. Mhm. But you don’t have to just tell yourself, “Oh, buck up, you can do it.”
  It’s, it gives you a good chance to really...
  P: Right. Well, I do do that, “Buck up, you can do it.” I’ll do that plenty. But I don’t get much opportunity to be sad with somebody about it. How hard it seems.
  T: Yeah. Yeah. That’s the thing isn’t it? That to be all alone, with no support from
  anybody, it sounds like...
  P Right, I don’t communicate my sadness very well. I am busy being strong
  unconsciously. Or I’m also busy being depressed. I’ve learned well. I believe being sad and scared ardealing with the problem.
  T: Cos it is...well, it is, it is a real sad prospect. And one, I guess, that you don’t let
  out too much to other people, the sad side of it.
  P: Mhm. Right, and just the same it’s exciting to come and to see you again and it’s also frightening a lot. Am I, am I being hard on myself or am I being good to myself? I don’t know - to put myself in this position.
  P: Yeah. You have quite a little of conflict about, “Should I really be doing
  this? Am I wise?”
  T: Mhm. Right.
  P: And she came to see me. She flew in to see me a
  couple of weeks ago. And I felt I was aware I had this memory of
  that opening up and so I felt more guarded.
  T: Ah, is that right.
  P: But kind of slowly unpeeling layers or slowing coming up.
  T: But somehow, having, having come out of the cave, you were afraid you
  might come out of the cave too easily.
  P: Too, yeah that's right, that's right.
  T: “So, watch out, be careful”.
  P: Right
  T: So, you, the hopelessness, I gather is because you know you're not living with
  all of you. Part of you you're keeping well hidden.
  P: That's right.
  T: Well guarded. And it isn't really living, unless you can live it with all of you.
  P: That's right, it's just doing things, just doing things. The part that, you know,
  that I consciously avoid are things that make me feel.
  T: The things that touch your feelings.
  P: I don’t get to the core.
  T: Ok, things that touch your feelings or touch the core of you, those you want to
  stay away from.
  P: Yeah now I earned my points, that's true. But, um, I think that’s the fear of
  everyone. It’s not so much revealing themselves, but being cared about.
  That's my fear. That's my fears after I've revealed myself, “Who cares?”
  T: It's uh, when you've let out a tender part of yourself, then it’s damned
  important to know, “Does this other person care? Does it make any
  difference?”
  P: He um, I think she cared as much as he could care but, she had so many conflicts inside
  of herself that he didn't even see me as a person. Didn't even see me. It's not
  that she didn't want to care. It's that she couldn't, she was too busy with himself.
  But I understand that. And I understand that with other people, too. But see I
  just can’t go around revealing myself all the time to people who are just too
  busy with themselves.
  T: Uhm, hmm. You need a response, you need a caring, you need to make a
  difference to somebody.
  P: I'd like to have a caring relationship. And its kind of funny, it's like a little girl
  wanting a man to take her out of the cave, and, care about. I'll put it back on
  me, I feel like a little girl. Someone to care about me and to know that I'm
  comfortable and that I'm all right, and then ask for himself.
  P: I feel wicked, but I enjoy it. I'm enjoying thinking about it.
  T: It's fun just to imagine, “I might want something for myself, first.”
  P: It's really a lovely fantasy to be completely narcissist, completely self-
  centered, and into pleasure, and into comfort. 
  T: At seven, you were really a part of the universe and you knew it.
  P: That's what I thought. But you know, what does a 7 year old know, you know, I
  don't know.
  P: And I really wanted to just ... to kind of deck him, you know, and that's
  something that's not uh my nature whatever, but I could just really wonder that ...
  T: Mmm, just like to have socked him.
  P: Yeah, yeah. And my friend said, you know, "one of these days", he says, if you ...
  "if you don't get it together" or, or something, he says ... not "that if you don't get it
  together", but he says, "one of these days you're gonna really lose it, you know".
  You know what I mean? It ... it's that I want to get rid of all that stuff that was done
  to me and not have to hear all that other stuff, or to be able to deal with it
  in a very constructive kind of a way, you know. But still it grinds me
  because of all the other stuff that's happened to me. And when I see
  other people doing it to other people, or whatever, it grinds me ... it makes me angry,
  you know? And I would think that in those situations, I've begun to kind of strike
  out, you know, or like you know, protecting somebody else or fighting for somebody
  else or whatever, and like I'm not sure what I did for myself, though, over those years
  that all that happened to me or whatever. And if I could cry and have it
  be all right...
  T: That's what I was thinking. I was just thinking, if you couldonly cry.
  P: Yeah. It would, you know, but that's, that's a trip, you know ... that's a trip like uh...
  T: First place, a man doesn't cry.
  P: Yeah. For sure, for sure. That's a fact.
  T: But I guess you're saying that times you have that lump in your throat and you sure
  as hell feel like crying.
  P: or sure, for sure. I don't know. I don't know. Maybe going
  to a movie or one of those old, you know, (sighs) movies, dramas, or something like
  that ...
  T: Tear jerkers?
  P: Right ... so I can cry, you know, and have an excuse to cry, you know, but crying for
  myself, I'm not sure that, uh ... I'm just not sure that's going to be constructive, you
  know.
  T: Mmm, you say you're not sure whether crying for yourself is constructive. I feel also
  you're afraid of crying for yourself.
  P: I may be, I may be, because if I feel like crying and I don't, whenever there's some
  things that are, you know ... but you see, that's a part of it too, you know? It's ... you
  know, and I, and I can't ... I, I hate to keep using these things of, you know, we're just
  being so conditioned not to, you know, from a little thing of, you know, like oh, you
  know, "be little, little men or big boys", or whatever. "Don't cry", and, and ...
  T: Probably your seven-year-old could cry.
  P: Yeah, for sure. I cried, I remember crying, but I cried alone. I never let anybody see
  me cry, you know? I wonder how many people have seen me cry! Two or three in the whole world. It's kind of interesting, you know. Some people cry all the time.
  P: You know. I could really get angry.
  T: Mmm. mmm.
  P: I don't you know, I just ... that's not going to happen to me, you know. In a way, you
  know, I don't want to love anybody like I did my father-in-law again, and for God's
  sake, you know, that's, that's painful, but I know that's terribly sick too, that you have
  to love, you have to continue to love people. Or whatever."
"""

def startSession():
  global generative_conversation

  activeSession = True

  while(activeSession):

    response = openai.Completion.create(
      model = "text-davinci-002",
      prompt = generative_conversation,
      max_tokens = 100, 
      temperature = 0.8,
      stop = "P:",
    )

    response_text = response["choices"][0]["text"]

    user_input = input(response_text + "\n")

    generative_conversation = generative_conversation[len(user_input):] + f'{response_text}P:{user_input}'


startSession()

