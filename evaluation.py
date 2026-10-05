"""Evaluate held-out recognition, retrieval coverage, and optional human review.

No model result is reported until a trained run exists. Coverage is not a
linguistic translation accuracy score. Human ratings must be collected separately.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from config import PROJECT_ROOT, RESULTS_DIR
from dataset_utils import load_split, file_hash
from translation import TextPreprocessor, SignVocabulary, plan_coverage

COVERAGE_SENTENCES = [
    'Hello, how are you?', 'Good morning, my friend.', 'I want to drink water.',
    'Thank you very much.', 'What is your name?', 'I am going to school today.',
    'Please help me with my homework.', 'The food is very sweet.', 'I love my family.',
    'Where is the market?', 'My mother is a teacher.', "It's raining cats and dogs.",
    'Can you give me a hand?', "I don't understand the question.",
    'We will travel to Lagos tomorrow.', 'The doctor said I should rest.',
    'How much does this cost?', 'I am very happy to see you.',
    'Please sit down and eat.', 'Good night, sleep well.']


def coverage_report():
    pp, vocab = TextPreprocessor(), SignVocabulary()
    rows = []
    for text in COVERAGE_SENTENCES:
        plan = vocab.resolve_tokens(pp.preprocess(text))
        rows.append({'sentence': text, **plan_coverage(plan),
                     'unavailable': [e['gloss'] for e in plan if e['type']=='missing']})
    total = sum(r['tokens'] for r in rows)
    direct = sum(r['direct_tokens'] for r in rows)
    playable = sum(r['direct_tokens']+r['fingerspelled_tokens'] for r in rows)
    return {'sentences': rows, 'token_count': total,
            'direct_coverage': direct/total if total else 0,
            'playable_coverage': playable/total if total else 0,
            'complete_sentences': sum(r['missing_tokens']==0 for r in rows),
            'note': 'Coverage measures availability, not NSL correctness or intelligibility.'}


def evaluate_run(run_dir, partition='test'):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from tensorflow.keras.models import load_model
    split_path = run_dir/'split_manifest.json'
    metadata = json.loads((run_dir/'run.json').read_text())
    if metadata['status'] != 'complete':
        raise ValueError('Training has not completed')
    if file_hash(split_path) != metadata['split_sha256']:
        raise ValueError('Saved split no longer matches the training run')
    split = json.loads(split_path.read_text())
    X, y = load_split(split, partition)
    model = load_model(run_dir/'best.keras')
    # Small direct batches avoid creating a separate tf.data thread pool.
    predicted = np.concatenate([np.asarray(model(X[i:i+16],training=False)).argmax(axis=1)
                                for i in range(0,len(X),16)])
    labels = list(range(len(split['label_map'])))
    names = [k for k,v in sorted(split['label_map'].items(),key=lambda x:x[1])]
    report = {'partition': partition, 'samples': len(y), 'classes': len(names),
              'accuracy': float(accuracy_score(y,predicted)),
              'classification': classification_report(y,predicted,labels=labels,target_names=names,
                                                        output_dict=True,zero_division=0),
              'limitation': split['limitation']}
    cm = confusion_matrix(y,predicted,labels=labels)
    np.save(run_dir/f'{partition}_confusion_matrix.npy',cm)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12,10))
    im=ax.imshow(cm,cmap='Blues'); fig.colorbar(im,ax=ax)
    ax.set(xlabel='Predicted class index',ylabel='Actual class index',title=f'{partition.title()} confusion matrix')
    fig.tight_layout(); fig.savefig(run_dir/f'{partition}_confusion_matrix.png',dpi=150); plt.close(fig)
    history_path=run_dir/'training_history.json'
    if history_path.exists():
        history=json.loads(history_path.read_text()); fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,key in zip(axes,['accuracy','loss']):
            ax.plot(history[key],label='train'); ax.plot(history['val_'+key],label='validation')
            ax.set(xlabel='Epoch (zero based)',ylabel=key); ax.legend()
        fig.tight_layout(); fig.savefig(run_dir/'training_curves.png',dpi=150); plt.close(fig)
    (run_dir/f'{partition}_metrics.json').write_text(json.dumps(report,indent=2))
    return report


def human_review_summary(path):
    rows=list(csv.DictReader(Path(path).open()))
    completed=[]
    for row in rows:
        if not row.get('reviewer_id') or not row.get('intelligibility_1_to_5'):
            continue
        ratings=[float(row[k]) for k in ['intelligibility_1_to_5','meaning_preservation_1_to_5','grammar_1_to_5']]
        if not all(1<=score<=5 for score in ratings):
            raise ValueError('Human ratings must be between 1 and 5')
        if row.get('fluent_nsl_user','').strip().lower()!='yes':
            continue
        completed.append(ratings)
    return {'submitted_rows':len(rows), 'completed_ratings':len(completed),
            'unique_fluent_reviewers':len({r['reviewer_id'] for r in rows
                 if r.get('reviewer_id') and r.get('fluent_nsl_user','').strip().lower()=='yes'}),
            'note':'Reviewer ratings are descriptive; report failures and disagreements separately.',
            'mean_intelligibility_meaning_grammar':np.mean(completed,axis=0).tolist() if completed else None}


def create_review_template(path):
    if path.exists():
        return
    with path.open('w',newline='') as handle:
        fields=['sentence_id','english_text','output_video','playback_mode','output_complete',
                'understood_text','reviewer_id','fluent_nsl_user',
                'intelligibility_1_to_5','meaning_preservation_1_to_5','grammar_1_to_5','notes']
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for i,text in enumerate(COVERAGE_SENTENCES,1):
            writer.writerow({'sentence_id':i,'english_text':text})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=PROJECT_ROOT/'models/baseline')
    parser.add_argument('--partition',choices=['validation','test'],default='test')
    parser.add_argument('--human-review',type=Path)
    args=parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    coverage=coverage_report()
    (RESULTS_DIR/'coverage.json').write_text(json.dumps(coverage,indent=2))
    create_review_template(RESULTS_DIR/'human_review.csv')
    report={'retrieval':coverage,'recognition':None,'human_review':None}
    if (args.run_dir/'best.keras').exists():
        report['recognition']=evaluate_run(args.run_dir,args.partition)
    if args.human_review:
        report['human_review']=human_review_summary(args.human_review)
    (RESULTS_DIR/'evaluation_report.json').write_text(json.dumps(report,indent=2))
    print(f"Direct coverage: {coverage['direct_coverage']:.1%}; playable: {coverage['playable_coverage']:.1%}; complete sentences: {coverage['complete_sentences']}/20")
    if report['recognition']:
        print(f"{args.partition} recognition accuracy: {report['recognition']['accuracy']:.1%}")
    else:
        print('Recognition not evaluated: no trained model in the selected run directory.')
    print('Human review template: results/human_review.csv (ratings are intentionally blank).')


if __name__=='__main__':
    main()
