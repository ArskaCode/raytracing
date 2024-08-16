use image::{Rgb32FImage, RgbImage, Rgb, ImageBuffer, buffer::ConvertBuffer, Pixel};
use rand::Rng;
use std::ops;
use rayon::iter::ParallelIterator;

#[derive(Copy, Clone, Debug)]
struct Vec3(f64, f64, f64);

#[derive(Debug)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}

enum Geometry {
    Sphere {
        center: Vec3,
        radius: f64,
    },
    Plane {
        y: f64,
    },
}

struct HitResult {
    position: Vec3,
    normal: Vec3,
    t: f64,
}

struct Material {
    albedo: Vec3,
    metalness: f64,
}

impl Vec3 {
    fn random(min: f64, max: f64) -> Vec3 {
        let mut rand = rand::thread_rng();
        Vec3(
            rand.gen_range(min..=max),
            rand.gen_range(min..=max),
            rand.gen_range(min..=max),
        )
    }

    fn random_unit() -> Self {
        loop {
            let vec = Self::random(-1.0, 1.0);
            if vec.sqr() <= 1.0 {
                return vec.unit();
            }
        }
    }

    fn cross(self, other: Vec3) -> Vec3 {
        Vec3(
            self.1 * other.2 - self.2 * other.1,
            self.0 * other.2 - self.2 * other.0,
            self.0 * other.1 - self.1 * other.0,
        )
    }

    fn elem_product(self, other: Vec3) -> Vec3 {
        Vec3(
            self.0 * other.0,
            self.1 * other.1,
            self.2 * other.2,
        )
    }

    fn dot(self, other: Vec3) -> f64 {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }

    fn sqr(self) -> f64 {
        self.dot(self)
    }

    fn mag(self) -> f64 {
        self.sqr().sqrt()
    }

    fn unit(self) -> Self {
        self / self.mag()
    }

    fn into_color(self) -> Rgb<f32> {
        Rgb([self.0 as f32, self.1 as f32, self.2 as f32])
    }
}

impl ops::Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Vec3(
            -self.0,
            -self.1,
            -self.2,
        )
    }
}

impl ops::Add for Vec3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Vec3(
            self.0 + other.0,
            self.1 + other.1,
            self.2 + other.2,
        )
    }
}

impl ops::Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Vec3(
            self.0 - other.0,
            self.1 - other.1,
            self.2 - other.2,
        )
    }
}


impl ops::Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        Vec3(
            self.0 * other,
            self.1 * other,
            self.2 * other,
        )
    }
}

impl ops::Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        Vec3(
            self.0 / other,
            self.1 / other,
            self.2 / other,
        )
    }
}

impl Ray {
    fn at(&self, t: f64) -> Vec3 {
        self.origin + self.direction*t
    }
}

impl Geometry {
    fn hit(&self, ray: &Ray, max_t: f64) -> Option<HitResult> {
        match self {
            Geometry::Sphere { center, radius } => {
                let oc = *center - ray.origin;
                let h = ray.direction.dot(oc);
                let c = oc.sqr() - radius*radius;
                
                let discrim = h*h - c;
                
                if discrim < 0.0 {
                    None
                } else {
                    let mut t = h - discrim.sqrt();
                    if t < 0.0001 || t >= max_t {
                        t = h + discrim.sqrt();
                        if t < 0.0001 || t >= max_t {
                            return None;
                        }
                    }
                    let position = ray.at(t);
                    let normal = (position - *center) / *radius;
                    Some(HitResult { position, normal, t })
               }
            }
            Geometry::Plane { y } => {
                if ray.direction.1 == 0.0 {
                    return None;
                }

                let t = (y - ray.origin.1) / ray.direction.1;
                
                if t < 0.0001 || t >= max_t {
                    None
                } else {
                    let position = ray.at(t);
                    let normal = Vec3(0.0, 1.0, 0.0);
                    Some(HitResult { position, normal, t })
                }
            }
        }
    }
}

fn reflect(vec: Vec3, normal: Vec3) -> Vec3 {
    vec - normal*2.0*vec.dot(normal)
}

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 1024;
const PIXEL_SUBDIVISIONS: usize = 20;

fn get_color(ray: Ray, depth: u32) -> Vec3 {
    if depth == 0 {
        return Vec3(0.0, 0.0, 0.0);
    }

    let sphere = Geometry::Sphere {
        center: Vec3(1.2, 0.0, -5.0),
        radius: 1.0,
    };

    let sphere2 = Geometry::Sphere {
        center: Vec3(0.0, -101.0, -5.0),
        radius: 100.0
    };

    let sphere3 = Geometry::Sphere {
        center: Vec3(-1.2, 0.0, -5.0),
        radius: 1.0
    };

    let sphere4 = Geometry::Sphere {
        center: Vec3(0.0, 1.8, -5.0),
        radius: 1.0
    };

    let _plane = Geometry::Plane {
        y: -1.0,
    };

    let sphere_material = Material {
        albedo: Vec3(0.9, 0.9, 0.9),
        metalness: 1.0,
    };

    let sphere2_material = Material {
        albedo: Vec3(0.5, 0.5, 0.5),
        metalness: 0.0,
    };

    let sphere3_material = Material {
        albedo: Vec3(0.9, 0.0, 0.0),
        metalness: 0.0,
    };

    let sphere4_material = Material {
        albedo: Vec3(0.8, 0.8, 0.0),
        metalness: 0.5,
    };

    let _plane_material = Material {
        albedo: Vec3(0.9, 0.9, 0.9),
        metalness: 1.0,
    };
    
    let world = [(sphere, sphere_material), (sphere2, sphere2_material), (sphere3, sphere3_material), (sphere4, sphere4_material)];

    let mut max_t = f64::INFINITY;
    let mut result = None;
   
    for (obj, mat) in world {
        let hit = obj.hit(&ray, max_t);
        hit.as_ref().inspect(|res| max_t = res.t);
        result = hit.map(|res| (res, mat)).or(result);
    }
     
    if let Some((result, mat)) = result {
        let p = rand::thread_rng().gen_range(0.0..=1.0);
        // Metalness represents the probability for the ray to reflect
        let scatter_vec = if p > mat.metalness {
            // Lambertian scattering
            (result.normal + Vec3::random_unit()).unit()
        } else {
            // Reflection
            reflect(ray.direction, result.normal)
        };

        let scatter_ray = Ray { origin: result.position, direction: scatter_vec };
        return get_color(scatter_ray, depth-1).elem_product(mat.albedo);
    } else {
        Vec3(0.9, 0.9, 0.9)
    }
}

fn main() {
    let mut buffer: Rgb32FImage = ImageBuffer::new(WIDTH, HEIGHT);

    let viewport_width = 1.0;
    let viewport_height = 1.0;
    let focal_length = 1.0;

    let camera_location = Vec3(0.0, 1.0, 0.0);
    let camera_direction = Vec3(0.0, 0.0, -1.0);
    let camera_up = Vec3(0.0, 1.0, 0.0);
    let camera_right = camera_direction.cross(camera_up);
  
    let left_upper = 
        camera_direction*focal_length
        - camera_right*(viewport_width/2.0)
        + camera_up*(viewport_height/2.0);

    buffer.par_enumerate_pixels_mut().for_each(|(x, y, pixel)| {
        let mut avg_color = Vec3(0.0, 0.0, 0.0);
        for sub_x in 0..PIXEL_SUBDIVISIONS {
            for sub_y in 0..PIXEL_SUBDIVISIONS {
                let x_offset = (sub_x as f64 + 0.5) / PIXEL_SUBDIVISIONS as f64;
                let y_offset = (sub_y as f64 + 0.5) / PIXEL_SUBDIVISIONS as f64;
                let direction = left_upper
                    + camera_right * viewport_width * (x as f64 + x_offset) / WIDTH as f64
                    - camera_up * viewport_height * (y as f64 + y_offset) / HEIGHT as f64;

                let ray = Ray {
                    origin: camera_location,
                    direction: direction.unit(),
                };
                let color = get_color(ray, 40);
                avg_color = avg_color + color / (PIXEL_SUBDIVISIONS*PIXEL_SUBDIVISIONS) as f64;
            }
        }
        *pixel = avg_color.into_color().map(f32::sqrt);
    });

    let converted: RgbImage = buffer.convert();

    converted.save("test.png").expect("Failed to save img");
}
