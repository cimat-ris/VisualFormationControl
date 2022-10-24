from numpy import pi

#p0=[[0.878282522914492,-0.175701001146909,-0.609958850671150],
#[-0.593222358497610,0.869103362509112,0.464602376389354],
#[1.05779225057201,1.20218398522248,1.67111112191539],
#[3.06954498638367,3.22583756532681,3.03741516869841],
#[-0.297407756063982,-0.241295395547807,0.129528359906631],
#[0.128523300338360,-0.0828366742289879,-0.143711364419188]]

#p0=[[0.717077385010136,0.818558766677466,0.164850361481461],
#[0.411199261124015,-0.418054434407827,-0.534061538358595],
#[1.20683013465449,1.05008462058959,1.43588319888689],
#[3.44759483663269,3.44016453715528,3.34989438915629],
#[-0.170743846852702,-0.301186338757143,0.00118755858017772],
#[0.0231523095702017,-0.140044949309631,0.105286732971113]]

## Para comparar puntos 3D con misma altura o diferentes, converge mejor para el primer caso
#p0=[[-0.00802794649331251,0.220148972487223,-0.656123127656434],
#[-0.625740540456770,0.388629734926281,0.723725126189405],
#[1.03076815580950,1.08073103408020,1.08963608587814],
#[3.58268331241888,2.89661309747880,3.14035815040242],
#[0.175785041830392,-0.384794238617617,0.791933760114449],
#[-0.581323829769684,0.785794763646152,0.494450555911336]]

#p0=[[-0.527776036584160,-0.810776122915001,-0.418597357961472],
#[-0.470395905506690,0.695735695508134,-0.814415838007346],
#[1.42853339659526,1.74730712212403,1.22233911586099],
#[3.12904860013233,2.19626226462842,3.20633683868037],
#[0.961601793456303,-0.261308400116402,0.618326396529785],
#[-0.272761326963508,-0.0748330357319734,0.535981045113011]]

##Pose que genera offset final en errores de consenso
#p0=[[0.340586105652014,-0.625719565855399,-0.708025013674948],
#[0.446672867082677,0.586470559481185,0.831416545539097],
#[1.25054159850250,1.33834243526006,0.804634224134068],
#[2.85104516437514,3.48796002713332,3.33351636388593],
#[-0.189209769785084,-0.294489035549644,0.221519437103379],
#[0.288563915188408,-0.0400181170143364,0.257397461730546]]

#p0=[[-0.296898857809793,-0.159157144265732,0.279167462975502],
#[-0.339718408147344,0.845839367524438,-0.357104966748215],
#[1.37069175463897,0.834427706779045,1.77874099048754],
#[3.67219146446333,2.60673251564890,2.78196891690444],
#[-0.143880901061464,-0.238680518636942,0.101124986797586],
#[-0.186701138527726,-0.481350985711011,1.06177624103507]]

## Pose que genera un error en rotación en la formación
#p0=[[0.123882589569947,-0.339812924319351,0.340586105652014],
#[-0.0550968460952295,0.0513596439111829,0.446672867082677],
#[0.811902069501241,0.965648729499781,1.25054159850250],
#[3.02788280830255,3.21278947979117,2.85104516437514],
#[-0.235841239740615,-0.165477260261891,-0.189209769785084],
#[0.205449366906135,0.107567503076977,0.288563915188408]]

## Poses para 3Cams con proyecciones fuera del FoV
#p0=[[0.817385015648184,0.0721904982468585,-0.771777770535749],
#[0.0770636035863921,-0.340002648712797,-0.572435159095761],
#[0.892988926870678,1.26348924776224,0.809332512027770],
#[3.43133539922503,2.81373972515701,2.88137525548998],
#[0.0996525331158200,-0.203526178837040,-0.343028625352567],
#[-0.348075160882491,-0.0314395617831643,0.158531468166292]]

#p0=[[0.873223583441657,0.700948036838241,-0.701635776201456],
#[0.367451582650142,-0.741044825826957,-0.887704494177208],
#[1.07173268019052,1.75101996634730,0.886652662812614],
#[2.79547198547147,2.85495846696875,3.31406637812567],
#[0.110104923058945,0.0455507493791993,-0.314364730149717],
#[-0.0139406843844571,-0.134710484736757,-0.219283983832674]]

#p0=[[0.890971206871619,0.456252187257138,0.521048088776977],
#[-0.598662572873876,-0.584932961114111,-0.510153138848233],
#[1.71762623311907,0.987769748432957,1.17627014133046],
#[3.06009121322362,2.89173690312456,2.87381833963619],
#[0.289907420668037,0.203757138795025,-0.345489917090521],
#[-0.324792043136814,-0.275652539639187,0.292094980995975]]

#p0=[[-0.763101967822848,-0.381770382740304,0.783400253670660],
#[-0.119095583384501,0.283878253998897,-0.176268388889027],
#[1.01308566439596,1.48403295340696,0.940046752025854],
#[3.14159265358979,3.14159265358979,3.14159265358979],
#[0,0,0],
#[-0.557357723986182,-0.185999519345062,-0.247926060550350]]

## # # # 4 agentes
#p0=[[0.248129975638902,0.422390482046562,-0.394254814749297,-0.788283864179022],
#[0.0692764265507727,0.0183569824598134,-0.339028357341272,-0.801453077258198],
#[1.45694599819336,1.50137045631825,0.935335250206770,0.985641036791162],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[0,0,0,0],
#[-1.03225547998869,0.236713209065598,-0.358213796252785,-1.23608743616749]]

## Falla con homografía
#p0=[[0.248129975638902,0.422390482046562,-0.394254814749297,-0.588283864179022],
#[0.0692764265507727,0.0183569824598134,-0.339028357341272,-0.501453077258198],
#[1.4,1.1,1.8,0.9],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[0,0,0,0],
#[-1.03225547998869,0.236713209065598,-0.358213796252785,-1.23608743616749]]

## p0(6,:)=zeros(1,4)]

## x-y-z formation
#p0=[[0.183305007620503,-0.638319688114183,-0.638319806089180,0.183304830709350],
#[0.0873764552968101,0.0873764965976021,-0.734248315100546,-0.734248359368001],
#[1.16195305548302,1.16195305128753,1.16195296793922,1.16195297213471],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[-2.39481310597896e-16,7.04407178298277e-16,1.91640475021965e-16,-8.41775920190693e-16],
#[-2.00284161207816e-15,-1.45432884135877e-15,-1.19218274389966e-15,-2.41208090080270e-15]]

#p0=[[0.183305007620503,-0.638319688114183,-0.638319806089180,0.183304830709350],
#[0.0873764552968101,0.0873764965976021,-0.734248315100546,-0.734248359368001],
#[1.5,1.5,1.5,1.5],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[-2.39481310597896e-16,7.04407178298277e-16,1.91640475021965e-16,-8.41775920190693e-16],
#[-2.00284161207816e-15,-1.45432884135877e-15,-1.19218274389966e-15,-2.41208090080270e-15]]

## x-y formation but different z's
#p0=[[0.183305007620503,-0.638319688114183,-0.638319806089180,0.183304830709350],
#[0.0873764552968101,0.0873764965976021,-0.734248315100546,-0.734248359368001],
#[1.4,1.0,1.2,1.3],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[-2.39481310597896e-16,7.04407178298277e-16,1.91640475021965e-16,-8.41775920190693e-16],
#[-2.00284161207816e-15,-1.45432884135877e-15,-1.19218274389966e-15,-2.41208090080270e-15]]

## y-z formation but different x's
#p0=[[0.8,-1.0,0.25,0.0],
#[0.0873764552968101,0.0873764965976021,-0.734248315100546,-0.734248359368001],
#[1.16195305548302,1.16195305128753,1.16195296793922,1.16195297213471],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[-2.39481310597896e-16,7.04407178298277e-16,1.91640475021965e-16,-8.41775920190693e-16],
#[-2.00284161207816e-15,-1.45432884135877e-15,-1.19218274389966e-15,-2.41208090080270e-15]]

## x-z formation but different y's
#p0=[[0.183305007620503,-0.638319688114183,-0.638319806089180,0.183304830709350],
#[0.0,1.3,-0.9,-0.5],
#[1.16195305548302,1.16195305128753,1.16195296793922,1.16195297213471],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[-2.39481310597896e-16,7.04407178298277e-16,1.91640475021965e-16,-8.41775920190693e-16],
#[-2.00284161207816e-15,-1.45432884135877e-15,-1.19218274389966e-15,-2.41208090080270e-15]]

## Same z's, different x-y
#p0=[[0.8,-1.0,0.25,0.0],
#[0.0,1.3,-0.9,-0.5],
#[1.16195305548302,1.16195305128753,1.16195296793922,1.16195297213471],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[-2.39481310597896e-16,7.04407178298277e-16,1.91640475021965e-16,-8.41775920190693e-16],
#[-2.00284161207816e-15,-1.45432884135877e-15,-1.19218274389966e-15,-2.41208090080270e-15]]

## Same x-y-z, different yaws
#p0=[[0.183305007620503,-0.638319688114183,-0.638319806089180,0.183304830709350],
#[0.0873764552968101,0.0873764965976021,-0.734248315100546,-0.734248359368001],
#[1.16195305548302,1.16195305128753,1.16195296793922,1.16195297213471],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[-2.39481310597896e-16,7.04407178298277e-16,1.91640475021965e-16,-8.41775920190693e-16],
#[0.3,-0.2,-0.1,0.25]]

## All x-y-z different
#p0=[[0.8,-1.0,0.25,0.0],
#[0.0,1.3,-0.9,-0.5],
#[1.4,1.0,1.2,1.3],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[-2.39481310597896e-16,7.04407178298277e-16,1.91640475021965e-16,-8.41775920190693e-16],
#[-2.00284161207816e-15,-1.45432884135877e-15,-1.19218274389966e-15,-2.41208090080270e-15]]

#p0=[[0.8,-1.0,0.25,0.0],
#[0.0,1.3,-0.9,-0.5],
#[1.4,0.8,1.2,1.6],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[-2.39481310597896e-16,7.04407178298277e-16,1.91640475021965e-16,-8.41775920190693e-16],
#[-2.00284161207816e-15,-1.45432884135877e-15,-1.19218274389966e-15,-2.41208090080270e-15]]

#All x-y-z-yaw different, the orientation of c4 defined the global final orientation
p0=[[0.8,-1.0,0.25,0.0],
[0.0,1.3,-0.9,-0.5],
[1.4,0.8,1.2,1.6],
[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
[0,0,0,0],
[0.2,-0.18,-0.3,-0.0]]

#p0=[[-0.390211486140397,-0.899678644586500,0.836428386015964,-0.238042442897516],
#[-0.780940292487725,-0.320062917563661,0.397483148448892,0.290679887822260],
#[1.74916826170964,1.17838323319508,1.10081636635616,1.41721906553473],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[0,0,0,0],
#[-0.162151480216542,0.162868398960869,-0.407611211733236,0.139263960191958]]

#p0=[[-0.329986461960506,0.837923542691867,-0.692851360155446,-0.0203871722587410],
#[-0.388226588092981,0.497601831957640,0.328817764104095,-0.0496011788359765],
#[1.77446683837197,1.55947152947854,1.39136917516276,1.39779169698388],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[0,0,0,0],
#[-0.466785294106381,-0.499512657068148,0.190607221322112,0.00779670106034292]]

#p0=[[-0.579222385529993,0.265891124101333,-0.643957878129734,0.329165352538104],
#[0.710776815152007,-0.269244821346285,0.455045022163952,-0.724225322700678],
#[1.66703164757401,1.43604679317145,0.929949197011700,0.879078737258219],
#[3.14159265358979,3.14159265358979,3.14159265358979,3.14159265358979],
#[0,0,0,0],
#[-0.192985348376653,0.389610024610922,-0.361335119993383,-0.346997634737635]]


#case 1
P=[[-0.5],
[-0.5],
[0  ]] 
#case 2
P=[[-0.5, -0.5],
[-0.5,  0.5],
[0,   0.2]] 
#case 3
P=[[0,   -0.5,  0.5],
[-0.5,  0.5, 0],
[0.0,  0.2, 0.3]]
#case 4
P=[[-0.5, -0.5, 0.5,  0.5],
[-0.5,  0.5, 0.5, -0.5],
# 0,    0.2, 0.3,  -0.1]]           
[0 ,   0.0, 0.0,  0.0]]
#case 5
P=[[-0.5, -0.5, 0.5, 0.5, 0.1],
[-0.5, 0.5, 0.5, -0.5, -0.3],
[0, 0.0, 0.0,  -0.0, 0.0]]
#0, 0.2, 0.3, -0.1, 0.1]]
#case 6
P=[[-0.5, -0.5, 0.5, 0.5, 0.1, -0.1],
[-0.5, 0.5, 0.5, -0.5, -0.3, 0.2],
[0, 0.0, 0.0, -0.0, 0.0, 0.0]]
#0, 0.2, 0.3, -0.1, 0.1, 0.15]]
#otherwise
P=[[-0.5, -0.5, 0.5, 0.5],
[-0.5, 0.5, 0.5, -0.5],
[0, 0.2, 0.3, -0.1]]

Ldown=180*pi/180;
    
# x1=[-0.5, -0.5, 1.2, Ldown+0*pi/180, 0*pi/180, -20*pi/180]';
# x2=[0.0, -0.0, 1.2, Ldown-0*pi/180, 0*pi/180, 20*pi/180]';
# x3=[0.7, -0.5, 1.2, Ldown+0*pi/180, 0*pi/180, 0*pi/180]';
# x1=[-0.5, -0.5, 1.0, Ldown+0*pi/180, 0*pi/180, 0*pi/180]';
# x2=[0.0, -0.0, 1.2, Ldown-0*pi/180, 0*pi/180, 0*pi/180]';
# x3=[-0.7, 0.5, 1.4, Ldown+0*pi/180, 0*pi/180, 0*pi/180]';
# x1=[-0.6, -0.6, 1.2, Ldown+0*pi/180, 0*pi/180, 0*pi/180]
# x2=[0.5, -0.0, 1.2, Ldown-0*pi/180, 0*pi/180, 0*pi/180]
# x3=[-0.6, 0.6, 1.2, Ldown+0*pi/180, 0*pi/180, 0*pi/180]
# pd=[x1, x2, x3]


