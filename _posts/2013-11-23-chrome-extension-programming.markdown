---
layout: post
comments: true
title:  "Chrome Extension Programming: Illustrating a Basic Survival Skill with a Twitter Case Study"
excerpt: "I illustrate a very valuable skill (Chrome Extension Programming) using a Twitter Case study. We will give Twitter a face lift, get it to refresh new tweets automatically, and highlight tweets from people who rarely tweet. All with a few lines of Javascript!"
date:   2013-11-23 20:00:00
---

### Extension Hacking
I wanted to share a few examples of a powerful skill that I've been gradually picking up over the last year. It is simply the ability to quickly hack together custom browser extensions in Chrome and using them to customize my favorite websites. Writing extensions is very fast: you need a short manifest file that contains some boring meta information, a few js/html files with your code in a folder, and then you simply activate the folder as an extension from the Extensions menu with a few clicks. In general, you can do a lot of fancy things with extensions:

- add buttons, context-menu items
- modify functionality of Omnibox
- create extension-specific webpages that display various data/settings
- your extension is allowed to have local (or synced), persistent storage of data
- you can run almost arbitrary Javascript over the DOM of any webpage

I can't stress how powerful the last item is. You can run Javascript. On top of any webpage. You can Read the page DOM. You can write to it, automatically, on load of the webpage or even periodically! This gives you complete freedom in modifying any webpage to your tastes: remove annoying content, add new features, log/scrape website data, change the layout, etc. It's completely crazy!

I'll walk you through some examples with possible mods of Twitter just to give you a glimpse of how easy and powerful this can be. Twitter is fun and I use it often, but their website is annoying, has some ugly elements, and sometimes lacks certain functionality I would like it to have. A normal person would request features and wait, but with the dark arts of extension hacking we can do much better. Lets get right to it.

### Fixing the Ugly

A recent change on Twitter added this ugly text, visible by default and always, on every single tweet:
<img src="/assets/chrome1.jpeg">

I understand it gets people to accidentally click on these more and pads Twitter's engagement numbers, but it's useless, ugly, and it just takes up too much space. Lets right click on one of these and choose Inspect Element. This opens up the HTML of the page and here we see the culprit DOM elements:

<img src="/assets/chrome2.jpeg">

So we have a list `<ul></ul>` with items `<li>` one for each of Reply, Retweet, Favorite and More. Inside every of them they have a `<a>`(anchor that processes the click action), deeper we have a `<span>` that becomes the icon, and finally followed by the ugly text wrapped in `<b>`. That looks easy enough, we will find all these elements based on their  class attribute, descend down to find the text and get rid of it.  So we create a new folder for TwitterClean extension, copy paste some manifest boring code and set it up to load a javascript file anytime twitter loads. For example, right after twitter.com page loads, lets execute:

```javascript
var clean_twitter = function(){
  var ugly = [];
  ugly.push('.action-reply-container');
  ugly.push('.action-rt-container');
  ugly.push('.action-del-container');
  ugly.push('.action-fav-container');
  ugly.push('.more-tweet-actions');

  for(var i=0;i&lt;ugly.length;i++) {
    var u = $(ugly[i]).find('b');
    u.text('');
  }
}
```

Load the Extension, refresh Twitter and poof! All the text is gone and we're just left with the icons. These suffice. Oh and while we're at code we run automatically on load of twitter.com, lets slip this one is as well:

```javascript
$('.promoted-tweet').hide(); // oops!
```

I'll let you figure out what that single naughty line of code does for you :)

### Loading new tweets automatically

Here's another annoyance: you have your Twitter running on your side monitor and new tweets come in, but Twitter doesn't load them automatically! It just shows this:

<img src="/assets/chrome3.jpeg">

That's the passive aggressive look of Twitter telling you that there are two more tweets to show, but also refusing to actually show them. That would be too useful to their users. Instead, they want you to stop what you're doing and click the button to load the new tweets. Luckily, you are skilled at extension hacking so you can simply right click the caption, go to Inspect Element, and see that the &lt;div&gt; element that tells you there are more tweets has class "js-new-tweets-bar". Easy enough:

```javascript
var periodic = function() {
  L = document.getElementsByClassName('js-new-tweets-bar');
  if(L.length > 0){
    L[0].click();
  }
}
setInterval(periodic, 1000);
```

When this gets run when **twitter.com** loads, it sets up the code to look for the annoying bar every second (1000 milliseconds) and then runs its click event handler which loads the new tweets. That's all it takes, and now your tweets are streaming down automatically whenever they are available without you having to explicitly refresh them all the time. We've only written code for 5 minutes and in that time we tweaked the way Twitter looked, removed some "functionality" and added some functionality! We're on a roll! Let's do something fancier now.

### Highlighting tweets from rare tweeters (wait, or tweepers?)

One day I decided to collect tweets on my timeline over a period of a week using Twitter's REST API and saw that 30 accounts make up 50% of everything I see on Twitter. Since I follow 384 accounts in total, that's only 7%! Unfortunately, for Twitter every tweet is created equal, which means that this annoying social media guru person who tweets 100 times a day completely drowns tweets coming from your other friends who believe that one should also have something worthy of tweeting too. Okay well it's not exactly like that but I wished there was a mechanism for highlighting the very infrequent tweeters and seeing that low frequency content. Twitter will never implement this because it makes Zero sense for their revenue model, but luckily, we can hack this together quite easily! First, here's a function that goes through all tweets on your timeline, looks at who tweeted, and "charges" every unique tweet to the originating user:

```javascript
var charge_tweets = function() {

  // get all tweets in twitter timeline
  var items = $('.tweet');
  for(var i=0;i&lt;items.length;i++) {
    var it = items[i];

    // extract information from tweet HTML
    var original_user = $(it).attr('data-screen-name');
    var retweeter = $(it).attr('data-retweeter');
    var tweet_id = $(it).attr('data-tweet-id');

    // a bit of logic
    var charged_user = original_user;
    if(typeof retweeter !== 'undefined') {
      charged_user = retweeter;
    }

    // charge tweet to the user
    if(charge.hasOwnProperty(charged_user)) {
      var L = charge[charged_user];
      if($.inArray(tweet_id, L) === -1) {
        L.push(tweet_id);  
      }
    } else {
      charge[charged_user] = [tweet_id];
    }
  }
};
```

Basically, it turns out every tweet has class "tweet", so it is trivial to iterate over them as seen above. Similarly, by inspecting the way the HTML is laid out, it turns out we can simply scrape the user and the (unique) tweet id and use it to build up a dictionary of `user_string -> [tweet id, ...]`. Of course, we will have to let this accumulate for a few days before it measures a good tweeting frequency distribution for all people we follow as we visit Twitter again and again always seeing new tweets from more people. But this also means we have to load and save the **charge** dictionary from Chrome's local extension storage or otherwise we would lose all our charging work whenever we close the Tab! Easy enough:

```javascript
var save_charge = function() {
  chrome.storage.local.set({'charge': charge});
}

var load_charge = function() {
  chrome.storage.local.get('charge', function (result) {
    if(result.charge) {
      charge = result.charge;
      console.log('loaded tweet frequency stats:');
      console.log(charge);
    } else {
      console.log('no tweet frequency to load');
    }
  });
}
```

Now we just make sure to run load_charge() at start up, and save_charge() anytime there are new tweets and our <em>charge</em> dictionary changes. Based on this <em>charge</em> dictionary we can easily find, say, the 50th percentile frequency, and highlight any tweet that comes from a user who tweets less often than 50% of the users we follow:

```javascript
var display_charges = function() {

  var items = $('.tweet');
  for(var i=0;i&lt;items.length;i++) {
    var it = items[i];

    // ... as above and then:

    var charged_tweets = charge[charged_user];
    var charge_count = charged_tweets.length;

    // adjust highlight color of the tweet according to rareness
    if(charge_percentile &gt; 0) {
      var ratio = charge_count / charge_percentile;
      var x = Math.floor(Math.min(ratio,1)*255);
      $(it).css('background-color', 'rgb(255,255,' + x + ')');
    }
  }
}
```

This is just one possibility out of many. Here, <em>ratio</em> will be low for users who rarely tweet, and we're setting their tweet to be yellow based on their rareness. Very hard to not notice on your timeline! :) And while we're at it, why not also fit in:

```javascript
var VIP = ['elonmusk'];
if($.inArray(charged_user, VIP) !== -1) {
  $(it).css('background-color', 'rgb(150,255,150)'); 
}
```

This way, Elon Musk's (or your other Twitter favorites) tweets will always glow a vibrant, green color that is hard to notice! Nice. Here's what we get:

<img src="/assets/chrome4.jpeg">

Just look at that! Mashable and some person who needed every single one of his followers to know "Aarrrgh" look normal, Elon's tweets are hard to miss green, and someone who doesn't tweet relatively as often is highlighted a bit as yellow.

### Summary

It took us ~100 lines and 10 minutes of Javascript (with a bit of practice) and we tweaked Twitter's look, removed err... undesirable content, made Twitter autorefresh, and added an entirely new feature that highlights infrequent tweepers!

Yet we've only barely scratched the surface. If you're  comfortable with navigating HTML of pages with Chrome's awesome inspector and writing HTML/Javascript/CSS, these quick hacks have the potential to significantly improve your online experience by giving you powerful options for customizing your favorite sites. And if you are not comfortable, perhaps it's time to head over to Chrome Extensions "<a href="http://developer.chrome.com/extensions/getstarted.html">Getting Started</a>" and write a few hacks :)

Oh, and if you'd like the full code of the above, you may find it here: <a href="http://cs.stanford.edu/people/karpathy/twitteropt.zip">LINK</a> (Note it is a bit rough around the edges, but then it is a quick hack after all!). Let me know if you have any issues on [@karpathy](https://twitter.com/karpathy), and until later!
