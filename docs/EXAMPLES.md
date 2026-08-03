# EmailGrammar — Illustrative Examples

Live output from the recommended config: **model=mini, beam=2, speller on, entity-protection on**. Regenerate with `python scripts/gen_examples.py`.

Rows tagged **Limitation** are included deliberately — they map to the roadmap items in the overview doc.


## Spelling (non-word typos)

**1. classic misspellings the speller fixes**

- in : `i recieved you're seperate messege yesterdey`
- out: `I received your separate message yesterday.`

**2. transpositions + doubled letters**

- in : `teh committe will discus it tommorow`
- out: `The committee will discuss it tomorrow.`


## Grammar (agreement / tense)

**3. subject-verb agreement**

- in : `the team have been working hardly on it`
- out: `The team has been working hard on it.`

**4. articles + preposition**

- in : `please find attach report which you was requesting`
- out: `Please find the attached report which you were requesting.`

**5. double negative + tense**

- in : `she dont know if the client will excepts our proposal`
- out: `She does know if the client will accept our proposal.`


## Spelling + grammar together

**6. the common email case**

- in : `i has recieve you're emails yesterday and wil responsd son`
- out: `I received your emails yesterday and will respond soon.`

**7. run-on-ish**

- in : `their are alot of thing we need too discus before the meating`
- out: `There are a lot of things we need to discuss before the meeting.`


## Entity protection: email

**8. address must stay byte-exact**

- in : `pls send the file to john.doe@rediff.com by eod`
- out: `plus send the file to john.doe@rediff.com by the end`


## Entity protection: URL

**9. link preserved**

- in : `the details is on https://rediff.com/offers/2026 kindly chek`
- out: `The details are on https://rediff.com/offers/2026 kindly check.`


## Entity protection: order id

**10. hyphenated id preserved**

- in : `my order id ORD-99213 hasnt arrive yet`
- out: `My order is ORD-99213 hast arrived yet.`


## Entity protection: amount + time

**11. money and time preserved**

- in : `the amt is Rs.4500 and meeting is at 3pm sharp`
- out: `The amount is Rs.4500 and the meeting is at 3pm sharp.`


## Entity protection: handle + domain

**12. mention + bare domain preserved**

- in : `cc rahul@team.io and visit team.io for detail`
- out: `cc rahul@team.io and visit team.io for details.`


## Entity protection: multi-entity

**13. 3 entities recovered by position, grammar still applied**

- in : `email me at a.b@x.com or visit https://x.com/h before 5pm`
- out: `email me at a.b@x.com or visit https://x.com/h before 5pm.`


## Casing / position

**14. stray full-stop merge, case kept**

- in : `That. Works.`
- out: `That works...`


## Limitation: model capacity

**15. 31M model gets the meaning wrong**

- in : `he dont have no time for finishing this projet by tommorow`
- out: `He has done no time to finish this project by tomorrow.`

**16. 'am gone' is locally valid -> no error signal (see overview 7.1)**

- in : `I am gone insane`
- out: `I am gone insane.`


## Limitation: speller is context-blind

**17. 'cup' becomes 'zip' (freq-ranked, no context)**

- in : `cant wait for the wirld cip finsl next weak`
- out: `I can't wait for the world zip final next week.`


## Limitation: lowercase name

**18. not regex-detectable -> can be mangled**

- in : `please give demra and raghu my best regard`
- out: `Please give me a debra and my best regards.`


## Limitation: shouted typo

**19. all-caps guarded -> speller leaves it for T5**

- in : `this is REALY URGENT pls RESPOND`
- out: `This is REALLY URGENT plus RESPOND.`


## Limitation: chat abbreviation

**20. 'pls' wrongly -> 'plus' (needs abbrev dict)**

- in : `pls revert asap thx`
- out: `plus revert asap the`
