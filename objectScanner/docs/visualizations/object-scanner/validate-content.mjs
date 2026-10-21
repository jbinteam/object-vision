import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {datasetDeepDive} from './deep-dataset.mjs';
import {poseDeepDive} from './deep-pose.mjs';
const root=fileURLToPath(new URL('../../../',import.meta.url));
const lessons=[...datasetDeepDive,...poseDeepDive];
assert.equal(lessons.length,12);
let sourceCount=0;
for(const [n,lesson] of lessons.entries()){
 assert.equal(lesson.id,`${n<6?'dataset':'pose'}-0${n%6+1}`);
 for(const field of ['summary','math','example'])assert.ok(lesson[field],`${lesson.id}: ${field}`);
 assert.ok(lesson.mechanism.length>=4,`${lesson.id}: missing mechanism`);
 assert.ok(lesson.settings.length>=3,`${lesson.id}: missing configuration`);
 assert.ok(lesson.checks.length>=3,`${lesson.id}: missing checks`);
 assert.ok(lesson.failures.length>=2,`${lesson.id}: missing failures`);
 assert.ok(lesson.sources.length>=2,`${lesson.id}: missing source evidence`);
 for(const s of lesson.sources){
  const lines=readFileSync(root+s.path,'utf8').split('\n').length;
  const refs=s.lines.match(/\d+/g)?.map(Number);
  assert.ok(refs?.length,`${s.path}: no line references`);
  assert.ok(refs.every(x=>x>=1&&x<=lines),`${s.path}: reference outside 1..${lines}`);
  sourceCount++;
 }
}
console.log(`Content verified: ${lessons.length} lessons, ${sourceCount} valid source references.`);
