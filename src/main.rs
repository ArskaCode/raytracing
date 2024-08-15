use image::{Rgb32FImage, RgbImage, Rgb, ImageBuffer, buffer::ConvertBuffer};
use std::ops;

#[derive(Copy, Clone, Debug)]
struct Vec3(f64, f64, f64);

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

impl Vec3 {
    fn cross(self, other: Vec3) -> Vec3 {
        Vec3(
            self.1 * other.2 - self.2 * other.1,
            self.0 * other.2 - self.2 * other.0,
            self.0 * other.1 - self.1 * other.0,
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

struct HitResult {
    position: Vec3,
    normal: Vec3,
}

impl Geometry {
    fn hit(&self, ray: &Ray) -> Option<HitResult> {
        match self {
            Geometry::Sphere { center, radius } => {
                let oc = ray.origin - *center;
                let h = ray.direction.dot(oc);
                let c = oc.sqr() - radius*radius;

                let discrim = h*h - c;

                if discrim < 0.0 {
                    None
                } else {
                    let t = h - discrim.sqrt();
                    let position = ray.direction*t + ray.origin;
                    let normal = (position - *center) / *radius;
                    Some(HitResult { position, normal })
               }
            }
            Geometry::Plane { y } => {
                if ray.direction.1 == 0.0 {
                    return None;
                }

                let t = (y - ray.origin.1) / ray.direction.1;
                
                if t < 0.0 {
                    None
                } else {
                    let position = ray.direction*t + ray.origin;
                    let normal = Vec3(0.0, 1.0, 0.0);
                    Some(HitResult { position, normal })
                }
            }
        }
    }
}
const WIDTH: u32 = 1024;
const HEIGHT: u32 = 1024;

fn get_normal_color(normal: Vec3) -> Rgb<f32> {
    let color_normal = normal / 2.0;
    Rgb([color_normal.0 as f32 + 0.5, color_normal.1 as f32 + 0.5, color_normal.2 as f32 + 0.5])
}

fn get_color(ray: Ray) -> image::Rgb<f32> {
    let sphere = Geometry::Sphere {
        center: Vec3(0.0, 0.0, -5.0),
        radius: 1.0,
    };

    let plane = Geometry::Plane {
        y: -1.0,
    };

    if let Some(result) = sphere.hit(&ray) {
        get_normal_color(result.normal)
    } else {
        if let Some(result) = plane.hit(&ray) {
            get_normal_color(result.normal)
        } else {
            Rgb([0.0, 0.0, 0.0])
        }
    }
}

fn main() {
    let mut buffer: Rgb32FImage = ImageBuffer::new(WIDTH, HEIGHT);

    let viewport_width = 1.0;
    let viewport_height = 1.0;
    let focal_length = 1.0;

    let camera_location = Vec3(0.0, 0.0, 0.0);
    let camera_direction = Vec3(0.0, 0.0, -1.0);
    let camera_up = Vec3(0.0, 1.0, 0.0);
    let camera_right = camera_direction.cross(camera_up);
  
    let left_upper = 
        camera_direction*focal_length
        - camera_right*(viewport_width/2.0)
        + camera_up*(viewport_height/2.0);
    
    for x in 0..WIDTH {
        for y in 0..HEIGHT {
            let direction = left_upper
                + camera_right * viewport_width * (x as f64 + 0.5) / WIDTH as f64
                - camera_up * viewport_height * (y as f64 + 0.5) / HEIGHT as f64;

            let ray = Ray {
                origin: camera_location,
                direction: direction.unit(),
            };
            buffer.put_pixel(x, y, get_color(ray));
        }
    }

    let converted: RgbImage = buffer.convert();

    converted.save("test.png").expect("Failed to save img");
}
