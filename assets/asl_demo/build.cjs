// Render letter-labelled hand illustrations; attribution is retained in README.
const fs = require('fs');
const sharp = require(process.argv[2]);
const source = fs.readFileSync(__dirname+'/source.svg','utf8');
const inner = source.slice(source.indexOf('>',source.indexOf('<svg'))+1,source.lastIndexOf('</svg>')).replace(/<metadata[\s\S]*?<\/metadata>/, '');
const rows = [
 ['ABCDEFG',[55,150,252,365,465,553,643,760],0,135],
 ['HIJKLM',[65,200,318,428,533,651,760],245,375],
 ['NOPQRS',[60,180,295,440,560,650,760],490,615],
 ['TUVWXYZ',[60,150,247,340,439,549,661,770],705,845]
];
(async()=>{
 for(const [letters,xs,y,end] of rows) for(let i=0;i<letters.length;i++){
  const letter=letters[i];
  const w=(xs[i+1]-xs[i])/4, h=(end-y)/4;
  const cropped=source.replace('viewBox="0 0 210 297"',`viewBox="${xs[i]/4} ${y/4} ${w} ${h}"`).replace('width="210mm"',`width="${w*6}"`).replace('height="297mm"',`height="${h*6}"`);
  const png=await sharp(Buffer.from(cropped)).png().toBuffer();
  const card=`<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="white"/><text x="320" y="48" text-anchor="middle" font-family="sans-serif" font-size="36" font-weight="bold">${letter}</text><image x="110" y="70" width="420" height="370" href="data:image/png;base64,${png.toString('base64')}"/></svg>`;
  await sharp(Buffer.from(card)).png().toFile(__dirname+'/'+letter+'.png');
 }
})();
