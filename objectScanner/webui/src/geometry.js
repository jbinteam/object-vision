import * as THREE from 'three';

export function frameMatrix(origin, rotationDeg) {
  if (![...origin, ...rotationDeg].every(Number.isFinite)) throw new Error('Frame values must be finite.');
  const rotation = new THREE.Euler(...rotationDeg.map(THREE.MathUtils.degToRad), 'ZYX');
  return new THREE.Matrix4().makeRotationFromEuler(rotation).setPosition(...origin);
}

// Keep gizmo poses in the same Rz · Ry · Rx convention as numeric inputs/export.
export function frameFromPose(position, quaternion) {
  const euler = new THREE.Euler().setFromQuaternion(quaternion, 'ZYX');
  return { origin: position.toArray(), rotation: [euler.x, euler.y, euler.z].map(THREE.MathUtils.radToDeg) };
}

export function frameFromPicks(origin, xPoint, yPoint) {
  const o = new THREE.Vector3(...origin);
  const x = new THREE.Vector3(...xPoint).sub(o);
  let y = new THREE.Vector3(...yPoint).sub(o);
  if (x.length() < 1e-9 || y.length() < 1e-9) throw new Error('Choose axis points away from the origin.');
  x.normalize(); y.normalize(); y.addScaledVector(x, -x.dot(y));
  if (y.length() < 1e-4) throw new Error('The three points are on one line. Choose a different +Y point.');
  y.normalize();
  const z = new THREE.Vector3().crossVectors(x, y).normalize();
  const matrix = new THREE.Matrix4().makeBasis(x, y, z);
  const euler = new THREE.Euler().setFromRotationMatrix(matrix, 'ZYX');
  return { origin: [...origin], rotation: [euler.x, euler.y, euler.z].map(THREE.MathUtils.radToDeg) };
}

export function inside(point, min, max) {
  return point.every((value, axis) => value >= min[axis] && value <= max[axis]);
}

export function cropGeometry(min, max, rotationDeg = [0,0,0], center = min.map((value,i)=>(value+max[i])/2)) {
  const half = new THREE.Vector3(...max.map((value,i)=>(value-min[i])/2));
  const matrix = frameMatrix(center,rotationDeg);
  return {matrix, inverse:matrix.clone().invert(), half};
}

export function moveCropFace(min, max, rotationDeg, center, side, axis, value) {
  const nextMin=[...min], nextMax=[...max];
  const bounds=side==='min'?nextMin:nextMax;
  const shift=new THREE.Vector3();shift.setComponent(axis,(value-bounds[axis])/2);
  bounds[axis]=value;
  if(!validCrop(nextMin,nextMax))throw new Error('Minimum must be below maximum on every box axis.');
  // Half of the face displacement moves the center; the opposite face stays put.
  shift.applyMatrix4(frameMatrix([0,0,0],rotationDeg));
  return {min:nextMin,max:nextMax,center:new THREE.Vector3(...center).add(shift).toArray()};
}

export function cropSliderBounds(sourceMin, sourceMax, min, max, rotationDeg, center) {
  const crop=cropGeometry(min,max,rotationDeg,center);
  const midpoint=new THREE.Vector3(...min.map((value,i)=>(value+max[i])/2));
  const box=new THREE.Box3();
  for(let corner=0;corner<8;corner++){
    const point=new THREE.Vector3(...sourceMin.map((value,i)=>(corner & (1<<i))?sourceMax[i]:value));
    box.expandByPoint(point.applyMatrix4(crop.inverse).add(midpoint));
  }
  return {min:box.min.toArray(),max:box.max.toArray()};
}

export function cropRecipe(min,max,rotationDeg,center) {
  // The server's recipe stores an equivalent box centered in scan coordinates,
  // then rotates it about that center. UI face values stay in the box's axes.
  const half=cropGeometry(min,max,rotationDeg,center).half.toArray();
  return {crop_min:center.map((value,i)=>value-half[i]),crop_max:center.map((value,i)=>value+half[i]),crop_rotation_deg:[...rotationDeg]};
}

export function validCrop(min, max) {
  return min.length === 3 && max.length === 3 && min.every((x, i) => Number.isFinite(x) && Number.isFinite(max[i]) && x < max[i]);
}

export function matrixRows(matrix) {
  const e = matrix.elements;
  return Array.from({length: 4}, (_, row) => Array.from({length: 4}, (_, column) => e[column*4+row]));
}
