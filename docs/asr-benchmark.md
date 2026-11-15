# ASR benchmark: Parakeet 110M vs streaming zipformer

Date: 2026-11-15 · machine: this laptop (CPU only, `onnxruntime` int8, 2 threads)

## Method

```bash
python tools/bench_asr.py --limit 73 --json /tmp/asr-bench-73.json
```

- Corpus: `hf-internal-testing/librispeech_asr_dummy` validation split, 73
  clips / 481 s of 16 kHz speech, reference transcripts provided.
- Same audio, same machine, same thread count for both models.
- WER/CER: corpus-level Levenshtein on uppercased, punctuation-folded text
  (so Parakeet's punctuation and capitals do not count as errors, nor does the
  zipformer's uppercase-only output).
- `RTF` = decode seconds / audio seconds. `First partial` = wall time until a
  non-empty streaming partial (zipformer only; Parakeet is offline).
- Each recognizer does one warm-up decode before measurement; `Load` is cold
  recognizer construction.

## Results

| Model | Clips | Audio | Load | Decode | RTF | First partial | WER | CER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zipformer | 73 | 481 s | 4.48 s | 62.1 s | 0.129 | 0.13 s | 9.2% | 4.5% |
| parakeet | 73 | 481 s | 1.77 s | 26.3 s | 0.055 | n/a (offline) | 4.6% | 1.6% |

Head to head: Parakeet is **2.0× more accurate** (4.6 % vs 9.2 % WER) and
**2.4× faster** (RTF 0.055 vs 0.129) on clean read speech, and it loads in half
the time. The zipformer's only remaining advantage is 0.13 s partials.

## Error examples

Clips where the normalised hypothesis differs from the reference (first five
per model); punctuation-only differences are excluded.


### zipformer — 59 of 73 clips differ

- `1272-128104-0000`
  - ref: MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
  - hyp: MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOS
- `1272-128104-0002`
  - ref: HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
  - hyp: HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MI
- `1272-128104-0003`
  - ref: HE HAS GRAVE DOUBTS WHETHER SIR FREDERICK LEIGHTON'S WORK IS REALLY GREEK AFTER ALL AND CAN DISCOVER IN IT BUT LITTLE OF ROCKY ITHACA
  - hyp: HE HAS GRAVE DOUBTS WHETHER SIR FREDERIC LAYTON'S WORK IS REALLY GREEK AFTER ALL AND CAN DISCOVER IN IT BUT LITTLE OF ROCKY ITHAC
- `1272-128104-0004`
  - ref: LINNELL'S PICTURES ARE A SORT OF UP GUARDS AND AT EM PAINTINGS AND MASON'S EXQUISITE IDYLLS ARE AS NATIONAL AS A JINGO POEM MISTER BIRKET FOSTER'S LANDSCAPES SMILE AT ONE MUCH IN THE SAME WAY THAT MISTER CARKER USED TO FLASH HIS TEETH AND MISTER JOHN COLLIER GIVES HIS SITTER A CHEERFUL SLAP ON THE BACK BEFORE HE SAYS LIKE A SHAMPOOER IN A TURKISH BATH NEXT MAN
  - hyp: LINEL'S PICTURES ARE A SORT OF UP GUARDS AND ATOM PAINTINGS AND MASONS EXQUISITE IDYLLS ARE AS NATIONAL AS A JINGO POEM MISTER BRICKETT FOSTER'S LANDSCAPES SMILE AT ONE MUCH IN THE SAME WAY THAT MISTER CARKER USED TO FLASH HIS TEETH AND MISTER JOHN CALLIERS GIVES HIS SITTER A CHEERFUL SLAP ON THE BACK BEFORE HE SAYS LIKE A SHAMPOOER IN A TURKISH BATH NEXT MAN
- `1272-128104-0005`
  - ref: IT IS OBVIOUSLY UNNECESSARY FOR US TO POINT OUT HOW LUMINOUS THESE CRITICISMS ARE HOW DELICATE IN EXPRESSION
  - hyp: IT IS OBVIOUSLY UNNECESSARY FOR US TO POINT OUT HOW LUMINOUS THESE CRITICISMS ARE HOW DELICATE AN EXPRESS

### parakeet — 30 of 73 clips differ

- `1272-128104-0001`
  - ref: NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER
  - hyp: Nor is mister Quilter's manner less interesting than his manner.
- `1272-128104-0004`
  - ref: LINNELL'S PICTURES ARE A SORT OF UP GUARDS AND AT EM PAINTINGS AND MASON'S EXQUISITE IDYLLS ARE AS NATIONAL AS A JINGO POEM MISTER BIRKET FOSTER'S LANDSCAPES SMILE AT ONE MUCH IN THE SAME WAY THAT MISTER CARKER USED TO FLASH HIS TEETH AND MISTER JOHN COLLIER GIVES HIS SITTER A CHEERFUL SLAP ON THE BACK BEFORE HE SAYS LIKE A SHAMPOOER IN A TURKISH BATH NEXT MAN
  - hyp: Linel's pictures are a sort of up guards and atom paintings, and Mason's exquisite idles are as national as a jingo poem. mister Burkitt Foster's landscapes smile at one much in the same way that mister Carker used to flash his teeth, and mister John Collier gives his sitter a cheerful slap in the back, before he says, like a shampoo in a Turkish bath next man
- `1272-128104-0006`
  - ref: ON THE GENERAL PRINCIPLES OF ART MISTER QUILTER WRITES WITH EQUAL LUCIDITY
  - hyp: Under general principles of art, mister Quilter writes with equal lucidity
- `1272-128104-0009`
  - ref: HE LAMENTS MOST BITTERLY THE DIVORCE THAT HAS BEEN MADE BETWEEN DECORATIVE ART AND WHAT WE USUALLY CALL PICTURES MAKES THE CUSTOMARY APPEAL TO THE LAST JUDGMENT AND REMINDS US THAT IN THE GREAT DAYS OF ART MICHAEL ANGELO WAS THE FURNISHING UPHOLSTERER
  - hyp: He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures, makes a customary appeal to the last judgment and reminds us that in the great days of art Michelangelo was the furnishing upholsterer
- `1272-128104-0010`
  - ref: NEAR THE FIRE AND THE ORNAMENTS FRED BROUGHT HOME FROM INDIA ON THE MANTEL BOARD
  - hyp: near the fire, and the ornaments Fred brought home from India on the mantle board.

## Caveats

- LibriSpeech is clean read English and Parakeet was trained on 36 k hours of
  English, so this measures the models on favourable input. The Parakeet card's
  own accented-speech test (EdACC) is the better guide for real users:
  streaming 38–45 % WER vs 13–17 % after re-decoding with this model.
- Single run on a laptop CPU; a phone will be slower but the ordering should
  hold (Parakeet is a 110 M transducer, the zipformer computes more per second
  of audio at RTF 0.129 vs 0.055).
- The benchmark needs `pyarrow` (parquet) and `ffmpeg` (flac → 16 kHz wav);
  neither is a runtime dependency of the app.
