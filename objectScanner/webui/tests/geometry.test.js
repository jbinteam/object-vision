import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {frameMatrix,frameFromPicks,frameFromPose,cropGeometry,moveCropFace,cropSliderBounds,cropRecipe,inside,validCrop,matrixRows} from '../src/geometry.js';

test('each rotated face moves along its own axis and leaves the opposite face fixed',()=>{
  const min=[1,-2,3],max=[5,2,5],rotation=[23,-41,67],center=[3,4,-2];
  const before=cropGeometry(min,max,rotation,center);
  for(const side of ['min','max'])for(let axis=0;axis<3;axis++){
    const delta=side==='min'?.4:-.4;
    const next=moveCropFace(min,max,rotation,center,side,axis,(side==='min'?min:max)[axis]+delta);
    const after=cropGeometry(next.min,next.max,rotation,next.center);
    const sign=side==='min'?-1:1;
    const face=(crop,s)=>new THREE.Vector3().setComponent(axis,s*crop.half.getComponent(axis)).applyMatrix4(crop.matrix);
    assert.ok(face(before,-sign).distanceTo(face(after,-sign))<1e-10,'opposite face drifted');
    const expected=new THREE.Vector3().setComponent(axis,delta).applyMatrix4(frameMatrix([0,0,0],rotation));
    assert.ok(face(after,sign).sub(face(before,sign)).distanceTo(expected)<1e-10,'face moved off its local axis');
    // Preserve the existing server crop recipe while serializing the shifted box.
    const recipe=cropRecipe(next.min,next.max,rotation,next.center);
    const exported=cropGeometry(recipe.crop_min,recipe.crop_max,recipe.crop_rotation_deg);
    exported.matrix.elements.forEach((value,i)=>assert.ok(Math.abs(value-after.matrix.elements[i])<1e-10));
    assert.ok(exported.half.distanceTo(after.half)<1e-10);
  }
});

test('a 90 degree box rotation makes its X minimum move along scan Y',()=>{
  const next=moveCropFace([-1,-1,-1],[1,1,1],[0,0,90],[0,0,0],'min',0,0);
  assert.ok(new THREE.Vector3(...next.center).distanceTo(new THREE.Vector3(0,.5,0))<1e-10);
  const recipe=cropRecipe(next.min,next.max,[0,0,90],next.center);
  const crop=cropGeometry(recipe.crop_min,recipe.crop_max,recipe.crop_rotation_deg);
  const kept=[[0,-.5,0],[0,.5,0]].map(point=>inside(new THREE.Vector3(...point).applyMatrix4(crop.inverse).toArray(),crop.half.clone().negate().toArray(),crop.half.toArray()));
  assert.deepEqual(kept,[false,true]);
});

test('slider limits follow rotated axes without jumping when a face moves',()=>{
  const sourceMin=[-3,-2,-1],sourceMax=[3,2,1],min=[-1,-1,-1],max=[1,1,1],rotation=[0,0,90],center=[0,0,0];
  const before=cropSliderBounds(sourceMin,sourceMax,min,max,rotation,center);
  before.min.forEach((value,i)=>assert.ok(Math.abs(value-[-2,-3,-1][i])<1e-10));
  const next=moveCropFace(min,max,rotation,center,'max',1,.2);
  const after=cropSliderBounds(sourceMin,sourceMax,next.min,next.max,rotation,next.center);
  for(const side of ['min','max'])after[side].forEach((value,i)=>assert.ok(Math.abs(value-before[side][i])<1e-10));
});

test('rotated crop selection matches the backend convention and rotates about its center',()=>{
  const crop=cropGeometry([-1.1,-.1,-1.1],[1.1,.1,1.1],[0,0,90]);
  const points=[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[2,2,2]];
  const selected=points.map(point=>inside(new THREE.Vector3(...point).applyMatrix4(crop.inverse).toArray(),crop.half.clone().negate().toArray(),crop.half.toArray()));
  assert.deepEqual(selected,[true,false,true,true,false]);
  const offset=cropGeometry([1,-2.3,3.5],[5,-1.7,4.5],[23,-41,67]);
  assert.ok(new THREE.Vector3().applyMatrix4(offset.matrix).distanceTo(new THREE.Vector3(3,-2,4))<1e-10);
  const local=new THREE.Vector3(.2,.1,-.3);
  assert.ok(local.clone().applyMatrix4(offset.matrix).applyMatrix4(offset.inverse).distanceTo(local)<1e-10);
});

test('dragged poses preserve the export transform, including pitch singularities',()=>{
  for(const rotation of [[23,-41,172],[40,90,-65],[-30,-90,120],[0,0,0]]){
    const expected=frameMatrix([.23,-.7,1.8],rotation);
    const position=new THREE.Vector3(),quaternion=new THREE.Quaternion();
    expected.decompose(position,quaternion,new THREE.Vector3());
    const frame=frameFromPose(position,quaternion);
    const actual=frameMatrix(frame.origin,frame.rotation);
    actual.elements.forEach((value,i)=>assert.ok(Math.abs(value-expected.elements[i])<1e-7));
    assert.deepEqual(frame.origin,[.23,-.7,1.8]);
  }
});

test('a 90 degree yaw maps object X to scan Y and origin maps correctly',()=>{
  const matrix=frameMatrix([1,2,3],[0,0,90]);
  const p=new THREE.Vector3(1,0,0).applyMatrix4(matrix);
  assert.ok(p.distanceTo(new THREE.Vector3(1,3,3))<1e-10);
  assert.ok(p.applyMatrix4(matrix.clone().invert()).distanceTo(new THREE.Vector3(1,0,0))<1e-10);
  assert.deepEqual(matrixRows(matrix).map(row=>row[3]),[1,2,3,1]);
});
test('three picked points create the same right-handed frame used by export',()=>{
  const f=frameFromPicks([1,2,3],[1,3,3],[0,2,3]);
  const matrix=frameMatrix(f.origin,f.rotation);
  assert.ok(Math.abs(matrix.determinant()-1)<1e-10);
  const z=new THREE.Vector3(0,0,1).transformDirection(matrix);
  assert.ok(z.distanceTo(new THREE.Vector3(0,0,1))<1e-10);
  assert.ok(Math.abs(f.rotation[2]-90)<1e-10);
});
test('ambiguous or invalid frames and crops are rejected',()=>{
  assert.throws(()=>frameFromPicks([0,0,0],[1,0,0],[2,0,0]));
  assert.throws(()=>frameFromPicks([0,0,0],[0,0,0],[0,1,0]));
  assert.throws(()=>frameMatrix([NaN,0,0],[0,0,0]));
  assert.equal(validCrop([0,0,0],[1,1,1]),true);
  assert.equal(validCrop([0,0,0],[0,1,1]),false);
  assert.equal(inside([0,1,.5],[0,0,0],[1,1,1]),true);
  assert.equal(inside([0,2,.5],[0,0,0],[1,1,1]),false);
});
